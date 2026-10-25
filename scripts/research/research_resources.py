"""Set bounded library threads and verify Windows process priority using typed handles."""

from __future__ import annotations

import ctypes
import os


def configure_background_job():
    for key in (
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ[key] = "1"
    result = {"library_thread_limit": 1}
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.SetPriorityClass.restype = ctypes.c_int
        kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
        kernel.GetPriorityClass.restype = ctypes.c_uint32
        process = kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(process, 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
        if kernel.GetPriorityClass(process) != 0x4000:
            raise RuntimeError("BelowNormal priority was not applied")
        result["windows_priority_verified"] = "BelowNormal"
    return result
