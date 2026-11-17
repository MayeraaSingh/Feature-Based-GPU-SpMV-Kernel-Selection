// SpMV benchmark runner: ONE matrix per process (a failure cannot contaminate other
// matrices).  Reads a binary matrix produced by scripts/export_gpu_inputs.py, verifies
// every kernel against the float64 CPU reference supplied in the file, times the
// kernels with CUDA events and writes raw per-repetition data.  All statistics
// (median, IQR, ...) are computed later in Python from the raw data.
//
// Timing protocol (see docs/benchmark_protocol.md):
//   * preprocessing: host wall clock around prepare()+bind() + cudaDeviceSynchronize,
//     repeated `prep_reps` times (release between), reported as raw samples.
//   * kernel time: each kernel's run() is captured once into a CUDA graph; each timed
//     repetition is  event(start) -> graph launch -> event(stop) -> synchronise.  Graph
//     replay removes host launch latency from the measurement.  Repetitions are
//     interleaved round-robin over the kernels (rotating start) to spread thermal and
//     clock drift evenly.  The L2 cache is NOT flushed (warm-cache protocol).
//   * correctness is checked on the first (cold) run with y pre-filled with NaN, and
//     again after all timed repetitions.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "../kernels/spmv_common.cuh"

using Clock = std::chrono::steady_clock;
static double wall_ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Problem {
  HostCsr csr;
  std::vector<float> x;
  std::vector<double> yref;
  double norm_a = 0, norm_x = 0;
};

static bool read_problem(const std::string& path, Problem& p, std::string& err) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open " + path; return false; }
  char magic[8];
  int64_t hdr[3];
  double nrm[2];
  bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, "SPMVBIN1", 8) == 0 &&
            fread(hdr, 8, 3, f) == 3 && fread(nrm, 8, 2, f) == 2;
  if (!ok) { fclose(f); err = "bad header"; return false; }
  p.csr.rows = (int)hdr[0]; p.csr.cols = (int)hdr[1]; p.csr.nnz = (int)hdr[2];
  p.norm_a = nrm[0]; p.norm_x = nrm[1];
  p.csr.rowptr.resize(p.csr.rows + 1);
  p.csr.colidx.resize(p.csr.nnz);
  p.csr.val.resize(p.csr.nnz);
  p.x.resize(p.csr.cols);
  p.yref.resize(p.csr.rows);
  ok = fread(p.csr.rowptr.data(), 4, p.csr.rows + 1, f) == (size_t)p.csr.rows + 1 &&
       fread(p.csr.colidx.data(), 4, p.csr.nnz, f) == (size_t)p.csr.nnz &&
       fread(p.csr.val.data(), 4, p.csr.nnz, f) == (size_t)p.csr.nnz &&
       fread(p.x.data(), 4, p.csr.cols, f) == (size_t)p.csr.cols &&
       fread(p.yref.data(), 8, p.csr.rows, f) == (size_t)p.csr.rows;
  fclose(f);
  if (!ok) err = "truncated file";
  return ok;
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct KernelState {
  std::unique_ptr<SpmvKernel> k;
  std::string name;
  std::string status = "ok";
  std::string error;
  std::string timing_mode = "graph";
  std::vector<double> prep_ms;
  bool correct_first = false, correct_last = false;
  double max_abs_err = 0, threshold = 0;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  int inner = 1;  // SpMVs per timed block; the reported time is block_time / inner
  std::vector<double> times;
};

__global__ void k_empty() {}

int main(int argc, char** argv) {
  std::map<std::string, std::string> a = {{"warmup", "20"}, {"reps", "100"}, {"eps", "1e-5"},
                                          {"prep-reps", "10"}, {"timing", "graph"},
                                          {"kernels", "csr_scalar,csr_vector,csr_adaptive,cusparse"},
                                          {"inject-fault", ""}, {"inner", "0"}, {"target-block-ms", "1.0"}};
  for (int i = 1; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
  const std::string id = a["id"], outdir = a["out"];
  if (id.empty() || outdir.empty() || a["matrix"].empty()) {
    fprintf(stderr, "usage: spmv_bench --matrix f.bin --id ID --out DIR [--warmup N --reps N "
                    "--eps E --prep-reps N --timing graph|direct --kernels a,b --inject-fault K]\n");
    return 2;
  }
  const int warmup = std::stoi(a["warmup"]), reps = std::stoi(a["reps"]);
  const int prep_reps = std::stoi(a["prep-reps"]);
  const double eps = std::stod(a["eps"]);
  const int inner_override = std::stoi(a["inner"]);
  const double target_block_ms = std::stod(a["target-block-ms"]);
  const bool use_graph = a["timing"] == "graph";
  const std::string base = outdir + "/" + id;

  std::ofstream fk(base + ".kernels.csv"), fr(base + ".raw.csv"), fa(base + ".aux.csv");
  fk << "kernel,status,error,timing_mode,correct_first,correct_last,max_abs_err,threshold,"
        "n_reps,warmup,fault_injected,inner_iters\n";
  fr << "phase,kernel,rep,ms\n";
  fa << "key,value\n";

  try {
    Problem p;
    std::string err;
    if (!read_problem(a["matrix"], p, err)) { fprintf(stderr, "input error: %s\n", err.c_str()); return 3; }
    const HostCsr& h = p.csr;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    size_t freeb = 0, totb = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totb));
    size_t need = (size_t)(h.rows + 1) * 4 + (size_t)h.nnz * 8 + (size_t)(h.cols + h.rows) * 4;
    fa << "gpu_name," << prop.name << "\ncc," << prop.major << "." << prop.minor
       << "\nmem_free_mb_before," << freeb / 1048576.0 << "\nmem_needed_mb_est," << need / 1048576.0
       << "\nrows," << h.rows << "\ncols," << h.cols << "\nnnz," << h.nnz << "\n";
    if ((double)need * 1.5 > (double)freeb) {
      fa << "skipped,insufficient_gpu_memory\n";
      fprintf(stderr, "skip %s: needs ~%.0f MiB, free %.0f MiB\n", id.c_str(), need / 1048576.0, freeb / 1048576.0);
      return 4;
    }

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // ---- allocation + host->device transfer (measured separately) ----
    DeviceCsr d;
    d.rows = h.rows; d.cols = h.cols; d.nnz = h.nnz;
    float *dx = nullptr, *dy = nullptr;
    auto t0 = Clock::now();
    CUDA_CHECK(cudaMalloc(&d.rowptr, (h.rows + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d.colidx, (size_t)std::max(h.nnz, 1) * 4));
    CUDA_CHECK(cudaMalloc(&d.val, (size_t)std::max(h.nnz, 1) * 4));
    CUDA_CHECK(cudaMalloc(&dx, (size_t)std::max(h.cols, 1) * 4));
    CUDA_CHECK(cudaMalloc(&dy, (size_t)std::max(h.rows, 1) * 4));
    CUDA_CHECK(cudaDeviceSynchronize());
    fa << "alloc_ms," << wall_ms(t0, Clock::now()) << "\n";
    std::vector<double> h2d;
    for (int r = 0; r < 5; ++r) {
      CUDA_CHECK(cudaEventRecord(e0, s));
      CUDA_CHECK(cudaMemcpyAsync(d.rowptr, h.rowptr.data(), (h.rows + 1) * 4, cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(d.colidx, h.colidx.data(), (size_t)h.nnz * 4, cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(d.val, h.val.data(), (size_t)h.nnz * 4, cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(dx, p.x.data(), (size_t)h.cols * 4, cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaEventRecord(e1, s));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      h2d.push_back(ms);
    }
    fa << "h2d_ms_median," << median(h2d) << "\n";

    // ---- timer floor: empty-kernel graph through the same event/graph path ----
    {
      cudaGraph_t g; cudaGraphExec_t ge;
      CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      k_empty<<<1, 32, 0, s>>>();
      CUDA_CHECK(cudaStreamEndCapture(s, &g));
      CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0ULL));
      std::vector<double> v;
      for (int r = 0; r < 120; ++r) {
        CUDA_CHECK(cudaEventRecord(e0, s));
        CUDA_CHECK(cudaGraphLaunch(ge, s));
        CUDA_CHECK(cudaEventRecord(e1, s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r >= 20) v.push_back(ms);
      }
      fa << "timer_floor_ms_median," << median(v) << "\n";
      cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }

    // ---- feature-gathering overhead: host pass vs device pass over rowptr ----
    {
      std::vector<double> vc, vg;
      double sink = 0;
      for (int r = 0; r < 55; ++r) {
        auto a0 = Clock::now();
        unsigned long long sq = 0; int mx = 0, mn = 0x7fffffff;
        for (int i = 0; i < h.rows; ++i) {
          int len = h.rowptr[i + 1] - h.rowptr[i];
          sq += (unsigned long long)len * len; mx = std::max(mx, len); mn = std::min(mn, len);
        }
        auto a1 = Clock::now();
        sink += (double)sq + mx + mn;
        if (r >= 5) vc.push_back(wall_ms(a0, a1));
        RowStats rs;
        auto b0 = Clock::now();
        gpu_row_stats(d, &rs, s);
        auto b1 = Clock::now();
        if (r >= 5) vg.push_back(wall_ms(b0, b1));
        if (r == 5) fa << "feature_gpu_matches_cpu," << ((rs.sumsq == sq && rs.maxlen == mx && rs.minlen == mn) ? 1 : 0) << "\n";
      }
      fa << "feature_cpu_ms_median," << median(vc) << "\nfeature_gpu_ms_median," << median(vg)
         << "\nsink," << sink << "\n";
    }

    // ---- build the candidate kernels ----
    std::vector<KernelState> ks;
    {
      std::string list = a["kernels"] + ",";
      size_t pos = 0;
      while (pos < list.size()) {
        size_t c = list.find(',', pos);
        std::string nm = list.substr(pos, c - pos);
        pos = c + 1;
        KernelState st;
        if (nm == "csr_scalar") st.k = make_csr_scalar();
        else if (nm == "csr_vector") st.k = make_csr_vector();
        else if (nm == "csr_adaptive") st.k = make_csr_adaptive();
        else if (nm == "csr_adaptive_multilaunch") st.k = make_csr_adaptive_multilaunch();
        else if (nm == "cusparse") st.k = make_cusparse();
        else { fprintf(stderr, "unknown kernel %s\n", nm.c_str()); return 2; }
        st.name = nm;
        ks.push_back(std::move(st));
      }
    }

    const double thr = eps * (p.norm_a * p.norm_x + 1.0);
    std::vector<float> yh(std::max(h.rows, 1));
    auto check = [&](KernelState& st, bool& flag) {
      CUDA_CHECK(cudaMemcpy(yh.data(), dy, (size_t)h.rows * 4, cudaMemcpyDeviceToHost));
      if (a["inject-fault"] == st.name && h.rows > 0) yh[0] += 1.0f;
      double mx = 0; bool finite = true;
      for (int i = 0; i < h.rows; ++i) {
        if (!std::isfinite(yh[i])) { finite = false; break; }
        mx = std::max(mx, std::fabs((double)yh[i] - p.yref[i]));
      }
      st.max_abs_err = finite ? std::max(st.max_abs_err, mx) : std::numeric_limits<double>::infinity();
      st.threshold = thr;
      flag = finite && mx <= thr;
    };

    for (auto& st : ks) {
      try {
        // warm the library / driver state (not timed, not reported)
        st.k->prepare(h, d, s); st.k->bind(dx, dy, s); CUDA_CHECK(cudaDeviceSynchronize()); st.k->release();
        for (int r = 0; r < prep_reps; ++r) {
          auto q0 = Clock::now();
          st.k->prepare(h, d, s);
          st.k->bind(dx, dy, s);
          CUDA_CHECK(cudaDeviceSynchronize());
          auto q1 = Clock::now();
          st.prep_ms.push_back(wall_ms(q0, q1));
          if (r + 1 < prep_reps) st.k->release();
        }
        // cold correctness run with poisoned output
        CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)std::max(h.rows, 1) * 4));  // NaN pattern
        st.k->run(dx, dy, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        check(st, st.correct_first);
        auto capture = [&](int K) {
          if (st.exec) { cudaGraphExecDestroy(st.exec); st.exec = nullptr; }
          if (st.graph) { cudaGraphDestroy(st.graph); st.graph = nullptr; }
          CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
          for (int q = 0; q < K; ++q) st.k->run(dx, dy, s);
          CUDA_CHECK(cudaStreamEndCapture(s, &st.graph));
          CUDA_CHECK(cudaGraphInstantiate(&st.exec, st.graph, 0ULL));
        };
        // Calibrate K so one timed block lasts about target_block_ms (fixed event/graph
        // overhead is then divided by K).  Pilot with K = 1 (median of 11 launches).
        int K = 1;
        if (inner_override > 0) {
          K = inner_override;
        } else {
          if (use_graph) capture(1);
          std::vector<double> pv;
          for (int q = 0; q < 16; ++q) {
            CUDA_CHECK(cudaEventRecord(e0, s));
            if (use_graph) CUDA_CHECK(cudaGraphLaunch(st.exec, s)); else st.k->run(dx, dy, s);
            CUDA_CHECK(cudaEventRecord(e1, s));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (q >= 5) pv.push_back(ms);
          }
          double t1 = std::max(median(pv), 1e-3);
          K = (int)std::min(1000.0, std::max(1.0, std::ceil(target_block_ms / t1)));
        }
        st.inner = K;
        if (use_graph) capture(K); else st.timing_mode = "direct";
      } catch (const std::exception& ex) {
        st.status = "failed";
        st.error = ex.what();
        for (auto& c : st.error) if (c == ',' || c == '\n' || c == '\r') c = ';';
        fprintf(stderr, "[%s] %s failed: %s\n", id.c_str(), st.name.c_str(), st.error.c_str());
        cudaGetLastError();
        cudaStreamSynchronize(s);
      }
    }

    auto launch = [&](KernelState& st) {
      if (use_graph) CUDA_CHECK(cudaGraphLaunch(st.exec, s));
      else for (int q = 0; q < st.inner; ++q) st.k->run(dx, dy, s);
    };
    for (int r = 0; r < warmup; ++r)
      for (auto& st : ks) if (st.status == "ok") { try { launch(st); } catch (...) { st.status = "failed"; st.error = "warmup"; } }
    CUDA_CHECK(cudaStreamSynchronize(s));

    const int K = (int)ks.size();
    for (int r = 0; r < reps; ++r) {
      for (int j = 0; j < K; ++j) {
        KernelState& st = ks[(j + r) % K];
        if (st.status != "ok") continue;
        try {
          CUDA_CHECK(cudaEventRecord(e0, s));
          launch(st);
          CUDA_CHECK(cudaEventRecord(e1, s));
          CUDA_CHECK(cudaEventSynchronize(e1));
          float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
          st.times.push_back((double)ms / st.inner);  // per-SpMV time
        } catch (const std::exception& ex) {
          st.status = "failed"; st.error = ex.what();
          for (auto& c : st.error) if (c == ',' || c == '\n' || c == '\r') c = ';';
        }
      }
    }

    // post-timing correctness (determinism / state corruption check)
    for (auto& st : ks) {
      if (st.status != "ok") continue;
      try {
        CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)std::max(h.rows, 1) * 4));
        launch(st);
        CUDA_CHECK(cudaStreamSynchronize(s));
        check(st, st.correct_last);
      } catch (const std::exception& ex) {
        st.status = "failed"; st.error = ex.what();
        for (auto& c : st.error) if (c == ',' || c == '\n' || c == '\r') c = ';';
      }
    }

    for (auto& st : ks) {
      fk << st.name << "," << st.status << "," << st.error << "," << st.timing_mode << ","
         << (st.correct_first ? 1 : 0) << "," << (st.correct_last ? 1 : 0) << "," << st.max_abs_err
         << "," << st.threshold << "," << st.times.size() << "," << warmup << ","
         << (a["inject-fault"] == st.name ? 1 : 0) << "," << st.inner << "\n";
      for (size_t i = 0; i < st.prep_ms.size(); ++i) fr << "prep," << st.name << "," << i << "," << st.prep_ms[i] << "\n";
      for (size_t i = 0; i < st.times.size(); ++i) fr << "kernel," << st.name << "," << i << "," << st.times[i] << "\n";
      if (st.exec) cudaGraphExecDestroy(st.exec);
      if (st.graph) cudaGraphDestroy(st.graph);
      st.k->release();
    }
    size_t freeb2 = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeb2, &totb));
    fa << "mem_free_mb_after_setup," << freeb2 / 1048576.0 << "\ntiming_mode," << (use_graph ? "graph" : "direct")
       << "\neps," << eps << "\nnorm_a_inf," << p.norm_a << "\nnorm_x_inf," << p.norm_x << "\n";
    cudaFree(d.rowptr); cudaFree(d.colidx); cudaFree(d.val); cudaFree(dx); cudaFree(dy);
    return 0;
  } catch (const std::exception& ex) {
    fprintf(stderr, "[%s] fatal: %s\n", id.c_str(), ex.what());
    fa << "fatal," << ex.what() << "\n";
    return 1;
  }
}
