param(
    [string]$RepoRoot = (Resolve-Path ".").Path
)

$ErrorActionPreference = "Stop"

$src = Join-Path $RepoRoot "facefx\runtime_cuda\native\facefx_runtime_cuda_native.cpp"
$outDir = Join-Path $RepoRoot "facefx\runtime_cuda\native"
$outDll = Join-Path $outDir "facefx_runtime_cuda_native.dll"

if (-not (Test-Path $src)) {
    throw "Source file not found: $src"
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe not found. Open 'x64 Native Tools Command Prompt for VS' and rerun."
}

Write-Host "Building native runtime CUDA helper DLL..."
& cl.exe /O2 /openmp /LD /EHsc /std:c++17 /Fe:$outDll $src

if (-not (Test-Path $outDll)) {
    throw "Build did not produce DLL: $outDll"
}

Write-Host "Wrote $outDll"
