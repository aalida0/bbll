#!/bin/bash
# HTCondor transferred launcher: Python code and JSON stay on shared AF filesystems.
# Explicit error checks; do not use set -e or set -u.
ERA="$1"
MASS="$2"
SHARD_INDEX="$3"
SHARDS="$4"
PYTHON="$5"
REPO="$6"
OUTPUT="$7"
MU_MAX="$8"

cd "$REPO"
if [ "$?" -ne 0 ]; then
  echo "Cannot access shared /home bbll repository: $REPO" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -c 'import pyhf; assert pyhf.__version__ == "0.7.6"'
if [ "$?" -ne 0 ]; then
  echo "pyhf==0.7.6 is unavailable in worker Python" >&2
  exit 3
fi

echo "era=$ERA mass=$MASS shard=$SHARD_INDEX/$SHARDS pyhf=$MU_MAX"
exec "$PYTHON" -m sr_optimization.exhaustive_shard \
  --era "$ERA" --mass "$MASS" \
  --shard-index "$SHARD_INDEX" --shards "$SHARDS" \
  --mu-max "$MU_MAX" --output-root "$OUTPUT"
