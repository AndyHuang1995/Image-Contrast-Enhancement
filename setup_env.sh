#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"
RECREATE=false

usage() {
    cat <<EOF
Usage: ./setup_env.sh [OPTIONS]

Set up a local Python virtual environment (.venv) and install dependencies.

Options:
  --python <path>   Specify the Python executable to use (requires Python >= 3.9)
  --recreate        Remove existing .venv and create a fresh environment
  -h, --help        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                echo "Error: --python requires a path argument." >&2
                exit 1
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        --recreate)
            RECREATE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

check_python_version() {
    local bin="$1"
    "$bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null
}

find_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            echo "Error: Specified Python executable not found: $PYTHON_BIN" >&2
            exit 1
        fi
        if ! check_python_version "$PYTHON_BIN"; then
            echo "Error: Python >= 3.9 is required, got: $("$PYTHON_BIN" --version 2>&1)" >&2
            exit 1
        fi
        echo "$PYTHON_BIN"
        return
    fi

    for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && check_python_version "$candidate"; then
            command -v "$candidate"
            return
        fi
    done

    echo "Error: Could not find Python >= 3.9. Please install Python 3.9+ or pass --python <path>." >&2
    exit 1
}

PYTHON_EXE="$(find_python)"
echo "==> Using Python: ${PYTHON_EXE} ($("${PYTHON_EXE}" --version))"

if [[ "$RECREATE" == true && -d "$VENV_DIR" ]]; then
    echo "==> Removing existing virtual environment: ${VENV_DIR}"
    rm -rf "$VENV_DIR"
fi

if command -v uv >/dev/null 2>&1; then
    echo "==> Detected uv; setting up virtual environment with uv..."
    if [[ ! -d "$VENV_DIR" ]]; then
        uv venv --python "$PYTHON_EXE" "$VENV_DIR"
    fi
    uv pip install --python "${VENV_DIR}/bin/python" -r "${SCRIPT_DIR}/requirements.txt"
else
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "==> Creating virtual environment at ${VENV_DIR}..."
        "$PYTHON_EXE" -m venv "$VENV_DIR"
    else
        echo "==> Reusing existing virtual environment at ${VENV_DIR} (use --recreate for a clean install)"
    fi

    echo "==> Upgrading pip..."
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip

    echo "==> Installing dependencies from requirements.txt..."
    "${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

echo "==> Verifying installed packages..."
"${VENV_DIR}/bin/python" - <<'PY'
import sys
import cv2
import imageio
import imageio.v3 as iio
import matplotlib
import numpy as np
import scipy
from scipy import optimize, sparse
from scipy.sparse.linalg import spsolve

assert hasattr(iio, "imread"), "imageio.v3.imread required"

print(f"  Python    : {sys.version.split()[0]}")
print(f"  numpy     : {np.__version__}")
print(f"  scipy     : {scipy.__version__}")
print(f"  opencv    : {cv2.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
print(f"  imageio   : {imageio.__version__}")
PY

cat <<EOF

Environment setup complete!
To activate the virtual environment, run:
  source .venv/bin/activate

Example commands:
  python ying.py testdata/01.jpg
  python dhe.py testdata/01.jpg
  python he.py testdata/01.jpg
  python ying.py testdata/01.jpg -o output.jpg --no-show
EOF
