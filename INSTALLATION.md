# Installation

This project uses `nvdiffrast`, which must compile a CUDA extension on Windows. The setup below is for Windows with PyTorch built for CUDA 12.8.

## Requirements

- 64-bit Python
- PyTorch built for CUDA 12.8
- NVIDIA CUDA Toolkit 12.8
- Visual Studio 2022 or Build Tools for Visual Studio 2022
- Visual Studio workload: Desktop development with C++
- MSVC v143 C++ x64/x86 build tools
- Windows 10 or Windows 11 SDK

## Check Python and PyTorch

Run:

```powershell
python -c "import platform, struct; print(platform.architecture()); print(struct.calcsize('P') * 8)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Python should report `64bit` and `64`. PyTorch should report CUDA `12.8`.

## Load the Visual Studio x64 Build Environment

From PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```

Confirm that the x64 compiler is available:

```powershell
where.exe cl
cl
```

`where.exe cl` should include a path like:

```text
VC\Tools\MSVC\...\bin\Hostx64\x64\cl.exe
```

## Select CUDA 12.8 for This Terminal

Run these commands in the same PowerShell session:

```powershell
$env:DISTUTILS_USE_SDK="1"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME=$env:CUDA_PATH
$env:CUDA_BIN_PATH=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
$env:NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
$env:MAX_JOBS="1"
```

Confirm that CUDA 12.8 is selected:

```powershell
where.exe nvcc
nvcc --version
```

`nvcc --version` should show CUDA release `12.8`.

## Install nvdiffrast

From the repo root, run:

```powershell
pip install -v git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation --no-cache-dir
```

## Verify Installation

Run:

```powershell
python -c "import nvdiffrast; print('nvdiffrast installed:', nvdiffrast.__file__)"
python -c "import torch; import nvdiffrast.torch as dr; print(torch.__version__, torch.version.cuda); print('OK')"
```

If the second command prints `OK`, `nvdiffrast` is installed for the active Python environment.

## Notes for Another Computer

The compiled `nvdiffrast` extension is not portable across machines in a normal repo checkout. Another computer needs the matching system setup at least once:

- 64-bit Python
- PyTorch CUDA 12.8 build
- CUDA Toolkit 12.8
- Visual Studio 2022 C++ build tools

The commands above make that setup repeatable.
