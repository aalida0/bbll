#!/usr/bin/env bash
# Read-only raw-Delphes FakeSim full-source worker, shared AF filesystem.
# Called as: full_worker.sh SAMPLE ORDINAL OUTPUT_PREFIX
# Do not use set -e / set -u; propagate Python's exit code explicitly.
if [ "$#" -ne 3 ]; then
    echo "usage: full_worker.sh SAMPLE ORDINAL OUTPUT_PREFIX" >&2
    exit 64
fi
SAMPLE="$1"
ORDINAL="$2"
PREFIX="$3"
cd /home/aalida/bbll || exit 65
unset PYTHONPATH
unset PYTHONHOME
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/home/aalida/bbll/.venv39/bin/python -m fake_backgrounds.full_shard \
  --manifest /data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json \
  --source-root /data/aalida/bbll_ml/fakes \
  --legacy-repo /home/aalida/bbll-ml-sr \
  --model-root /data/aalida/bbll_rerun_20261114 \
  --grid /data/aalida/bbll_rerun_20261114/fake_backgrounds/run3_candidate_grid.json \
  --output-root /data/aalida/bbll_rerun_20261114 \
  --sample "$SAMPLE" \
  --ordinal "$ORDINAL" \
  --output-prefix "$PREFIX" \
  --alpha-nodes 8 \
  --chunk-size 500
RESULT="$?"
if [ "$RESULT" -ne 0 ]; then
    echo "FAKESIM_FULL_SHARD_FAILED sample=$SAMPLE ordinal=$ORDINAL rc=$RESULT" >&2
    exit "$RESULT"
fi
echo "FAKESIM_FULL_SHARD_PASS sample=$SAMPLE ordinal=$ORDINAL"
exit 0
