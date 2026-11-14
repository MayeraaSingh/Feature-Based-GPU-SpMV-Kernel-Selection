"""
build_gpu.py
============
Configure and build the CUDA benchmark (gpu/ -> gpu/build/spmv_bench[.exe]).

Windows: initialises the x64 MSVC environment through VsDevCmd.bat (toolset selected
with --msvc-toolset, default 14.44, which is the one verified with CUDA 13.1), then
uses CMake + Ninja.  Linux/macOS: calls cmake directly.

    python scripts/build_gpu.py [--msvc-toolset 14.44] [--arch native|86] [--clean]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU = ROOT / "gpu"
BUILD = GPU / "build"


def find_cuda() -> Path | None:
    env = os.environ.get("CUDA_PATH")
    if env and Path(env, "bin").is_dir():
        return Path(env)
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.is_dir():
        versions = sorted(base.glob("v*"))
        if versions:
            return versions[-1]
    nvcc = shutil.which("nvcc")
    return Path(nvcc).parent.parent if nvcc else None


def find_vsdevcmd() -> Path | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.is_file():
        out = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        for line in out:
            p = Path(line) / "Common7/Tools/VsDevCmd.bat"
            if p.is_file():
                return p
    for guess in (r"C:\Program Files\Microsoft Visual Studio\18\Community",
                  r"C:\Program Files\Microsoft Visual Studio\2022\Community"):
        p = Path(guess) / "Common7/Tools/VsDevCmd.bat"
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msvc-toolset", default="14.44")
    ap.add_argument("--arch", default="native", help="CMAKE_CUDA_ARCHITECTURES (native, 86, ...)")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    cuda = find_cuda()
    if cuda is None:
        print("ERROR: CUDA Toolkit not found (set CUDA_PATH). See docs/environment_setup.md.", file=sys.stderr)
        return 2
    if args.clean and BUILD.exists():
        shutil.rmtree(BUILD)

    cmake_args = f'-S "{GPU}" -B "{BUILD}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES={args.arch}'
    if os.name == "nt":
        vs = find_vsdevcmd()
        if vs is None:
            print("ERROR: Visual Studio (VsDevCmd.bat) not found.", file=sys.stderr)
            return 2
        cmake_dir = vs.parents[1] / "IDE/CommonExtensions/Microsoft/CMake"
        extra = ";".join([str(cuda / "bin"), str(cuda / "bin" / "x64"),
                          str(cmake_dir / "CMake/bin"), str(cmake_dir / "Ninja")])
        import tempfile
        lines = [
            "@echo off",
            f'call "{vs}" -arch=x64 -host_arch=x64 -vcvars_ver={args.msvc_toolset} >nul',
            f'set "PATH={extra};%PATH%"',
            f"cmake {cmake_args} -G Ninja -DCMAKE_CUDA_HOST_COMPILER=cl || exit /b 1",
            f'cmake --build "{BUILD}" || exit /b 1',
        ]
        with tempfile.TemporaryDirectory() as td:
            bat = Path(td) / "build_gpu.bat"
            bat.write_text("\r\n".join(lines) + "\r\n")
            rc = subprocess.run(["cmd", "/c", str(bat)]).returncode
    else:
        rc = subprocess.run(f"cmake {cmake_args} && cmake --build {BUILD}", shell=True).returncode
    exe = BUILD / ("spmv_bench.exe" if os.name == "nt" else "spmv_bench")
    if rc != 0 or not exe.is_file():
        print(f"ERROR: build failed (exit {rc}).", file=sys.stderr)
        return rc or 1
    print(f"built {exe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
