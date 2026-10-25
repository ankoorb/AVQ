#!/bin/bash
# chmod +x run_generation.sh
# Usage: ./run_generation.sh "<temperatures>" "<experiment ids>" "<model seeds>"
# Examples:
#   ./run_generation.sh "1.0"    # T=1.0, ids 0-9, seed 42
#   ./run_generation.sh "1.0" "0 1 2 3 4 5 6 7 8 9" "42 259 2027"   # three model seeds
#   ./run_generation.sh "0.8 0.9 1.0 1.1" "0 1 2 3 4 5 6 7 8 9" "42"   # temperature sweep, seed 42
#   ./run_generation.sh "1.0" "10 11 12 13" "42"   # Mini-Places ids, passed explicitly

cd /workspace/AVQ || exit 1
python -m compileall -q avq

PROJECT_ROOT=/workspace/project
GFID_DIR=/workspace/gfid
TEMPS=${1:-"1.0"}
IDS=${2:-"0 1 2 3 4 5 6 7 8 9"}   # Tiny-ImageNet default
SEED=${3:-"42"}                   # model seed (42 is the default in generate/gfid)

mkdir -p logs

for temp in $TEMPS; do
  failed=0
  for id in $IDS; do
    echo "=== $(date '+%F %T') | model_seed=$SEED | temperature=$temp | experiment_id=$id ==="
    python -m avq.generation.generate \
        --project_root "$PROJECT_ROOT" \
        --experiment_id "$id" \
        --model_seed "$SEED" \
        --sampling_temperature "$temp" \
      && python -m avq.generation.gfid \
        --project_root "$PROJECT_ROOT" \
        --experiment_id "$id" \
        --model_seed "$SEED" \
        --output_dir "$GFID_DIR" \
      || { echo "FAILED model_seed=$SEED temperature=$temp experiment_id=$id" | tee -a logs/failed.txt; failed=1; }
  done
done
