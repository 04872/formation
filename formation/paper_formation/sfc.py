"""User-approved legacy GCOPTER FIRI with a conservative 2-D slice."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
from .config import RADIUS


class SFCConstructionError(RuntimeError):
    """The FIRI region cannot safely contain the current robot anchor."""


@lru_cache(maxsize=1)
def _function():
    library = ctypes.CDLL(str(Path(__file__).parent / 'native/libgcopter_firi.so'))
    function = library.gcopter_firi
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [pointer, ctypes.c_int, pointer, ctypes.c_double,
                         ctypes.c_double, ctypes.c_int, pointer, ctypes.c_int]
    function.restype = ctypes.c_int
    return function


def generate_sfc(blocked_points: np.ndarray, resolution: float, anchor: np.ndarray, inflate: bool = True):
    """Inflate around the point seed, intersect z=0, then shrink for the car disk."""
    halfwidth, halfheight = 2.0, 0.5
    margin = RADIUS + resolution*np.sqrt(2)/2
    # Include all cells whose inflated disk can intersect the bounded xy slice.
    delta = blocked_points-anchor
    selected = np.all(np.abs(delta) <= halfwidth+margin, axis=1)
    points = np.ascontiguousarray(blocked_points[selected], dtype=np.float64)
    anchor = np.ascontiguousarray(anchor, dtype=np.float64)
    if np.any(np.linalg.norm(points-anchor, axis=1) <= margin):
        raise SFCConstructionError('current anchor intersects an inflated blocked cell')
    output = np.empty((len(points)+6,4),dtype=np.float64)
    ptr = ctypes.POINTER(ctypes.c_double)
    count = _function()(points.ctypes.data_as(ptr),len(points),anchor.ctypes.data_as(ptr),
                        halfwidth,halfheight,4,output.ctypes.data_as(ptr),len(output))
    if count < 0:
        raise SFCConstructionError(f'GCOPTER FIRI returned {count}')
    planes=output[:count]
    h=planes[:,:2].copy()
    b=-planes[:,3]-margin*np.linalg.norm(h,axis=1)
    # z-only constraints are constants on the z=0 slice and do not limit x,y.
    planar=np.linalg.norm(h,axis=1)>1e-12
    h,b=h[planar],b[planar]
    if np.any(h@anchor>b+1e-9):
        raise SFCConstructionError('shrunk FIRI region excludes the current anchor')
    # Check every local blocked grid disk is excluded, including numerical tolerance.
    norms=np.linalg.norm(h,axis=1)
    if len(points) and np.any(np.max((points@h.T-b)/norms,axis=1)<margin-1e-8):
        raise SFCConstructionError('FIRI output does not separate all blocked cell disks')
    return h,b
