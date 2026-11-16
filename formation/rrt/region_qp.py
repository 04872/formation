"""ctypes binding of ``region_qp.c``; the shared library is (re)built with gcc when missing or older than the source."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).with_name("region_qp.c")
_LIBRARY = Path(__file__).with_name("_region_qp.so")


def _load() -> ctypes.CDLL:
    if not _LIBRARY.exists() or _LIBRARY.stat().st_mtime < _SOURCE.stat().st_mtime:
        subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", str(_LIBRARY), str(_SOURCE), "-lm"], check=True)
    library = ctypes.CDLL(str(_LIBRARY))
    vec = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.region_qp_project.argtypes = [vec, vec, ctypes.c_int, vec, ctypes.c_int, vec, vec]
    library.region_qp_project.restype = ctypes.c_int
    library.region_qp_margin.argtypes = [vec, vec, ctypes.c_int, vec, ctypes.c_int, vec,
                                         ctypes.POINTER(ctypes.c_double)]
    library.region_qp_margin.restype = ctypes.c_int
    return library


_LIB = _load()


def project(a: np.ndarray, b: np.ndarray, cones: np.ndarray, p: np.ndarray) -> tuple[int, np.ndarray]:
    """``argmin 1/2 ||z - p||^2`` over ``a z <= b`` and cones rows ``(cx, cy, ct, s, g)``:
    ``||z_xy - c_xy|| <= g - s (z_t - c_t)``.  Status 1 solved, 0 no interior point, -1 numerical failure."""
    z = np.empty(3)
    status = _LIB.region_qp_project(a, b, len(b), cones, len(cones), p, z)
    return status, z


def margin(a: np.ndarray, b: np.ndarray, cones: np.ndarray) -> tuple[int, float, np.ndarray]:
    """``max tau`` with a point at slack ``>= tau`` in every constraint, and that point."""
    z = np.empty(3)
    tau = ctypes.c_double()
    status = _LIB.region_qp_margin(a, b, len(b), cones, len(cones), z, ctypes.byref(tau))
    return status, tau.value, z
