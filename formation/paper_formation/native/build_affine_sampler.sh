#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-/home/eai/projects/env-rebuilt/bin/python}
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NUMPY_INCLUDE=$($PYTHON -c 'import numpy; print(numpy.get_include())')
PYTHON_INCLUDE=$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
NUMPY_LIBRARY=$($PYTHON -c 'import numpy; from pathlib import Path; print(Path(numpy.__file__).parent / "random/lib/libnpyrandom.a")')
cc -O3 -std=c11 -shared -fPIC -ffp-contract=off -I"$NUMPY_INCLUDE" -I"$PYTHON_INCLUDE" "$DIR/affine_sampler.c" "$NUMPY_LIBRARY" -lm -o "$DIR/libaffine_sampler.so"
