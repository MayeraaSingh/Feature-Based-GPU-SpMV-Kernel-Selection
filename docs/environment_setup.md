# Environment Setup

Verified on the project's reference machine (Windows 11, RTX 3050 6GB Laptop GPU, compute capability 8.6).

| Component | Version used | Notes |
|---|---|---|
| NVIDIA driver | 592.82 | not modified |
| CUDA Toolkit | 13.1 (nvcc 13.1.115) | `winget install Nvidia.CUDA --version 13.1`; installs to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1` |
| cuSPARSE | 12.7.3 (API version 12703) | shipped with the toolkit |
| MSVC host compiler | toolset 14.44 (cl 19.44.35228) | from Visual Studio 2026 Community; selected with `VsDevCmd.bat -vcvars_ver=14.44`. 14.51 was not tested |
| CMake / Ninja | 4.3.1 / bundled | bundled with Visual Studio (`...\Common7\IDE\CommonExtensions\Microsoft\CMake`) |
| Python | 3.13.7 | numpy 2.3.3, scipy 1.16.2, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.6 |

CUDA 13 on Windows keeps its runtime DLLs in `bin\x64`. The installer adds that directory to the machine `PATH`, but
terminals opened before the installation do not see it (a program then exits with `0xC0000135`). Open a new terminal;
`gpu_bench/runner.py` also prepends the directory for the benchmark subprocess.

## Commands
```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py        # read-only report; exit code 0 = ready for GPU work
python scripts/build_gpu.py                # configures with CMake + Ninja and builds gpu/build/spmv_bench.exe
```
CPU-only work (selector training, evaluation, tests that do not touch the GPU) needs only the Python packages.
GPU tests are skipped, with an explicit reason, when the executable or an NVIDIA GPU is absent; they are never replaced
by CPU emulation.

## Known environment limitations
* Nsight Compute is installed but cannot read GPU performance counters without permission
  (`ERR_NVGPUCTRPERM`). Enabling it needs administrator rights (NVIDIA Control Panel -> Desktop -> Enable Developer
  Settings -> Manage GPU Performance Counters -> allow all users). Profiling was therefore not used; kernel analysis
  relies on timings and `ptxas -v` resource output.
* The GPU is a laptop part under WDDM. Clocks are not locked (and are never changed by this project); telemetry is logged.
