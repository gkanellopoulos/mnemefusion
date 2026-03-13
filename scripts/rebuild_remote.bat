@echo off
REM Rebuild mnemefusion Python bindings with CUDA support
set LLAMA_BUILD_SHARED_LIBS=1
set CMAKE_CUDA_ARCHITECTURES=86
set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cd /d %~dp0..\mnemefusion-python
echo Environment:
echo   LLAMA_BUILD_SHARED_LIBS=%LLAMA_BUILD_SHARED_LIBS%
echo   CMAKE_CUDA_ARCHITECTURES=%CMAKE_CUDA_ARCHITECTURES%
echo   PYO3_USE_ABI3_FORWARD_COMPATIBILITY=%PYO3_USE_ABI3_FORWARD_COMPATIBILITY%
echo.
echo Starting build...
maturin develop --release --features entity-extraction-cuda
if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo BUILD SUCCEEDED
