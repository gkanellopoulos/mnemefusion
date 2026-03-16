import os as _os
import pathlib as _pathlib

# Set MNEMEFUSION_DLL_DIR so the native backend loader can find ggml-cpu.so
# and ggml-cuda.so bundled in mnemefusion.libs/ (pip wheel layout).
if "MNEMEFUSION_DLL_DIR" not in _os.environ:
    _libs_dir = _pathlib.Path(__file__).parent.parent / "mnemefusion.libs"
    if _libs_dir.is_dir():
        _os.environ["MNEMEFUSION_DLL_DIR"] = str(_libs_dir)

from .mnemefusion import *

__doc__ = mnemefusion.__doc__
if hasattr(mnemefusion, "__all__"):
    __all__ = mnemefusion.__all__
