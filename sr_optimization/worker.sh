#!/bin/bash
# UChicago AF: transfer only the launcher; use worker-visible /home and /data.
# No set -e or set -u; each failure is checked explicitly.
ERA="$1"
MASS="$2"
SHORTLIST="$3"
PYTHON="$4"
REPO="$5"
OUTPUT="$6"

cd "$REPO"
if [ "$?" -ne 0 ]; then
  echo "Cannot reach shared bbll repository: $REPO" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -m bdt_training.environment
if [ "$?" -ne 0 ]; then
  echo "Python runtime verification failed" >&2
  exit 3
fi

"$PYTHON" -c 'import pyhf; assert pyhf.__version__ == "0.7.6"'
if [ "$?" -ne 0 ]; then
  echo "pyhf 0.7.6 is missing or the wrong version" >&2
  exit 4
fi

exec "$PYTHON" -m sr_optimization.compare_architectures \
  --era "$ERA" --mass "$MASS" --shortlist "$SHORTLIST" \
  --output-root "$OUTPUT"
