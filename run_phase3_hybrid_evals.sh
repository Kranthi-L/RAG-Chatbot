#!/usr/bin/env bash
set -euo pipefail

# Phase 3 hybrid evaluations helper
# Runs run_batch_eval with hybrid retriever across courses and temperatures.

COURSES=("networking" "architecture" "machine_learning")
TEMPS=("0.2" "0.5" "0.8")
TOPK=8

# cd to script directory so paths are stable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for course in "${COURSES[@]}"; do
  for temp in "${TEMPS[@]}"; do
    echo "=== Phase3: course=${course}, temp=${temp}, top_k=${TOPK}, retriever=hybrid ==="
    python -m eval.run_batch_eval \
      --course "${course}" \
      --retriever_type hybrid \
      --temperature "${temp}" \
      --top_k ${TOPK} \
      --learner_level none \
      --max_workers 1
  done
done

echo "Phase 3 hybrid eval runs completed."
