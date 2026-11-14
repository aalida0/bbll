#!/bin/bash
# HTCondor transfers this small launcher to the worker's scratch directory.
# Python, the repository and selected ROOT inputs must remain accessible
# through their absolute /home and /data paths, as in the legacy AF workflow.

ERA="$1"
TOPOLOGY="$2"
PYTHON="$3"
REPO="$4"
SELECTED="$5"
OUTPUT="$6"

cd "$REPO"
if [ "$?" -ne 0 ]; then
  echo "Cannot access repository on worker: $REPO" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -m bdt_training.environment
if [ "$?" -ne 0 ]; then
  echo "Training Python environment failed verification" >&2
  exit 3
fi

exec "$PYTHON" -m bdt_training.train "$ERA" "$TOPOLOGY" \
  --selected-root "$SELECTED" --output-root "$OUTPUT"
