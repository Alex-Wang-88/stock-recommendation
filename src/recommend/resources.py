"""Bound unattended work without changing CPU affinity or desktop focus."""

from __future__ import annotations

import ctypes
import os


def background_limits(workers: int = 6) -> int:
    workers = max(1, min(int(workers), 6))
    for name in ("OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(workers)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    return workers
