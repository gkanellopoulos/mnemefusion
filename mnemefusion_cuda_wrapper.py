#!/usr/bin/env python3
"""
MnemeFusion CUDA Wrapper

Pre-loads CUDA runtime DLLs before importing mnemefusion to ensure
GPU-accelerated inference works correctly.

Usage:
    from mnemefusion_cuda_wrapper import setup_cuda, mnemefusion
    setup_cuda()
    memory = mnemefusion.Memory(...)
"""

import os
import sys
import ctypes
from ctypes import windll
from pathlib import Path

# CUDA Installation Path
CUDA_PATH = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")
CUDA_BIN_X64 = CUDA_PATH / "bin" / "x64"


def setup_cuda() -> bool:
    """
    Pre-load CUDA runtime DLLs to prepare for mnemefusion import.

    Returns:
        True if CUDA was successfully loaded, False if CPU fallback will be used
    """

    if not CUDA_BIN_X64.exists():
        print(f"[WARNING] CUDA path not found: {CUDA_BIN_X64}")
        print("[INFO] Will attempt CPU-only import")
        return False

    print(f"[CUDA] Setting up from: {CUDA_BIN_X64}")

    # Add CUDA path to DLL search path (Windows 10+)
    try:
        dll_add = windll.kernel32.AddDllDirectory
        dll_add.argtypes = [ctypes.c_wchar_p]
        dll_add.restype = ctypes.c_void_p
        cookie = dll_add(str(CUDA_BIN_X64))
        if cookie:
            print("[OK] CUDA DLL directory added to search path")
        else:
            print("[WARNING] AddDllDirectory returned null (may still work)")
    except Exception as e:
        print(f"[WARNING] AddDllDirectory failed: {e}")

    # Pre-load critical CUDA DLLs in dependency order
    critical_dlls = [
        "cudart64_13.dll",      # CUDA Runtime - must load first
        "cublas64_13.dll",       # cuBLAS
        "cublasLt64_13.dll",     # cuBLAS-LT
    ]

    loaded_dlls = []
    failed_dlls = []

    for dll_name in critical_dlls:
        dll_path = CUDA_BIN_X64 / dll_name

        if not dll_path.exists():
            print(f"[WARNING] DLL not found: {dll_name}")
            continue

        try:
            # Load with full path
            handle = ctypes.CDLL(str(dll_path))
            loaded_dlls.append(dll_name)
            print(f"[OK] Loaded: {dll_name}")
        except OSError as e:
            failed_dlls.append((dll_name, str(e)))
            print(f"[ERROR] Failed to load {dll_name}: {e}")

    print()

    if loaded_dlls:
        print(f"[SUCCESS] {len(loaded_dlls)}/{len(critical_dlls)} critical DLLs loaded")
        return True
    else:
        print("[FAILURE] No CUDA DLLs loaded - CPU-only fallback")
        return False


def import_mnemefusion():
    """
    Import mnemefusion after CUDA setup.

    Returns:
        The mnemefusion module

    Raises:
        ImportError if mnemefusion cannot be imported
    """
    try:
        import mnemefusion as mf
        print("[OK] mnemefusion imported successfully")
        return mf
    except ImportError as e:
        print(f"[ERROR] Failed to import mnemefusion: {e}")
        raise


# Module-level setup: attempt CUDA initialization on import
print("=" * 70)
print("MnemeFusion CUDA Wrapper - Initializing...")
print("=" * 70)

cuda_success = setup_cuda()

try:
    mnemefusion = import_mnemefusion()
    if cuda_success:
        print()
        print("[STATUS] Ready for GPU-accelerated inference")
    else:
        print()
        print("[STATUS] Running in CPU-only mode")
except ImportError:
    print()
    print("[FAILURE] mnemefusion module not available")
    raise

print("=" * 70)
print()

# Export the main module for convenience
__all__ = ['mnemefusion', 'setup_cuda', 'CUDA_PATH', 'CUDA_BIN_X64']
