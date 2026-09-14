#!/usr/bin/env bash

set -e

SCRIPT_PATH="${BASH_SOURCE:-$0}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)"
ENVIRONMENT_NAME="${PATHOMENU_ENV_NAME:-PathoMENU}"
PYTHON_VERSION="${PATHOMENU_PYTHON_VERSION:-3.10}"
TORCH_VERSION="${PATHOMENU_TORCH_VERSION:-2.4.1}"
CUDA_VERSION="${PATHOMENU_CUDA_VERSION:-cu124}"
PYG_TORCH_VERSION="${PATHOMENU_PYG_TORCH_VERSION:-2.4.0}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found."
    return 1 2>/dev/null || exit 1
fi

eval "$(conda shell.posix hook)"
if ! conda env list | awk -v name="${ENVIRONMENT_NAME}" '$1 == name { found = 1 } END { exit !found }'; then
    conda create -n "${ENVIRONMENT_NAME}" python="${PYTHON_VERSION}" -y
fi
conda activate "${ENVIRONMENT_NAME}"
python -m pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
python -m pip install torch-scatter==2.1.2 --only-binary=:all: -f "https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+${CUDA_VERSION}.html"
python -m pip uninstall -y esm fair-esm >/dev/null 2>&1 || true
python -m pip install fair-esm==2.0.0
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
chmod +x "${PROJECT_ROOT}/tools/foldx" "${PROJECT_ROOT}/tools/foldseek"
