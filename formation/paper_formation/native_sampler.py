"""Exact NumPy-bitgenerator affine candidate sampling via a small C library."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def _function():
    library=ctypes.CDLL(str(Path(__file__).parent/'native/libaffine_sampler.so'))
    f=library.sample_affine
    ptr=ctypes.POINTER(ctypes.c_double)
    f.argtypes=[ctypes.c_void_p,ctypes.c_double,ptr,ptr,ctypes.c_int,ctypes.c_double,ctypes.c_int,ptr]
    f.restype=ctypes.c_uint64
    return f


def sample_affine(rng, direction, matrix, geometry, count):
    """Return accepted angle/radius/matrix draws and number of attempts."""
    matrix=np.ascontiguousarray(matrix,dtype=np.float64).reshape(4)
    reference=np.ascontiguousarray(geometry.leader_reference,dtype=np.float64)
    output=np.empty((count,6),dtype=np.float64)
    ptr=ctypes.POINTER(ctypes.c_double)
    with rng.bit_generator.lock:
        attempts=_function()(rng.bit_generator.ctypes.bit_generator,float(direction),
              matrix.ctypes.data_as(ptr),reference.ctypes.data_as(ptr),len(reference),
              geometry.pair_min,count,output.ctypes.data_as(ptr))
    return output,attempts
