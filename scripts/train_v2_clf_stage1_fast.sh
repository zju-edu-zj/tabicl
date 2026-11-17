#!/usr/bin/env bash
set -eo pipefail
# One-command launcher for the measured node. Overrides work as in stage1.sh.
# SDK activation must precede importing vendor torch. Standard CUDA machines
# skip this block; override PPU_SDK_SETUP if installed elsewhere.
SDK_SETUP=${PPU_SDK_SETUP:-/usr/local/PPU_SDK/envsetup.sh}
if [[ -f "$SDK_SETUP" ]]; then
    source "$SDK_SETUP" >/dev/null
fi
set -u
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x /opt/conda/bin/python ]]; then
        export PYTHON=/opt/conda/bin/python
    else
        export PYTHON=python
    fi
fi
# Console tools such as ninja belong to the selected Python environment too.
# A fresh SSH login on the measured node otherwise finds /usr/bin/python and
# cannot find the ninja installed next to /opt/conda/bin/python.
if [[ "$PYTHON" == */* ]]; then
    export PATH="$(dirname "$PYTHON"):$PATH"
fi
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export MAX_JOBS=${MAX_JOBS:-1}
export TRAIN_DTYPE=${TRAIN_DTYPE:-bfloat16}
exec bash "$REPO_ROOT/scripts/train_v2_clf_stage1.sh"
