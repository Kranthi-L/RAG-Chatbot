#!/usr/bin/env bash
set -euo pipefail

# Courses in your project
COURSES=("networking" "architecture" "machine_learning")

# Per-course temperature lists
TEMPS_networking=("0.7" "1.0")          # resume set
TEMPS_architecture=("1.0")   # original set
TEMPS_machine_learning=("0.0" "0.2" "0.5" "0.7" "1.0") # original set

# Per-course top-k lists
TOPKS_networking=("2")                       # resume set
TOPKS_architecture=("2" "4" "6" "8")                 # original set
TOPKS_machine_learning=("2" "4" "6" "8")             # original set

# Limit parallel workers inside run_batch_eval.py
#export MAX_WORKERS=2


cd "$(dirname "$0")"

for course in "${COURSES[@]}"; do
  # pick temp/topk list for this course
  temps_var="TEMPS_${course}[@]"
  topks_var="TOPKS_${course}[@]"
  temps=("${!temps_var}")
  topks=("${!topks_var}")

  for temp in "${temps[@]}"; do
    for k in "${topks[@]}"; do
      echo "=== Running course=${course}, temp=${temp}, top_k=${k} ==="
        python run_batch_eval.py \
            --course "${course}" \
            --temperature "${temp}" \
            --top_k "${k}"
      
    done
  done
done

echo "All eval runs completed."
