#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
TARGET=${1:-"${REPO_ROOT}/../env-rebuilt"}

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    printf 'ERROR: this rebuild script supports Linux x86_64 only (got %s/%s).\n' "$(uname -s)" "$(uname -m)" >&2
    exit 2
fi
if [[ -e "${TARGET}" ]]; then
    printf 'ERROR: target already exists; refusing to modify it: %s\n' "${TARGET}" >&2
    exit 3
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA="${CONDA_EXE}"
elif CONDA=$(command -v conda); then
    :
else
    printf 'ERROR: conda was not found; set CONDA_EXE or add conda to PATH.\n' >&2
    exit 4
fi
if [[ ! -x "${CONDA}" ]]; then
    printf 'ERROR: conda executable is not executable: %s\n' "${CONDA}" >&2
    exit 4
fi

LOCK_FILE="${REPO_ROOT}/conda-linux-64.lock"
PIP_FILE="${REPO_ROOT}/requirements-pip-linux-64.txt"
if [[ ! -f "${LOCK_FILE}" || ! -f "${PIP_FILE}" ]]; then
    printf 'ERROR: rebuild inputs are missing under %s.\n' "${REPO_ROOT}" >&2
    exit 5
fi

printf 'Creating exact Conda environment at %s\n' "${TARGET}"
"${CONDA}" create --yes --prefix "${TARGET}" --file "${LOCK_FILE}"

PYTHON="${TARGET}/bin/python"
printf 'Installing pip-only distributions without dependency resolution\n'
"${PYTHON}" -m pip install --no-deps --requirement "${PIP_FILE}"

printf 'Checking installed package metadata\n'
"${PYTHON}" -m pip check

printf 'Checking core imports\n'
"${PYTHON}" - <<'PYTHON_CHECK'
import casadi
import cvxpy
import matplotlib
import numpy
import pandas
import scipy
import shapely
import PySide6
import shiboken6
import qdldl
import osqp
print('core imports: OK')
PYTHON_CHECK

printf 'Running focused tests\n'
PYTHONPATH="${REPO_ROOT}" "${PYTHON}" -m pytest -q tests/test_maps.py tests/test_tube_rrt.py
printf 'Environment rebuild validation completed successfully.\n'
