#!/usr/bin/env bash
# Screening ablation grid launcher (local).
#
# Grid: classes {4 smallest, 9 mid, 8 largest-minority}
#     x configs {v1+log_minmax (baseline), v2+log_minmax (circuit effect),
#                v2+quantile (circuit+preproc)}
#     x seeds {42, 43}
#     @ 50 epochs, patience 6  -> 18 jobs
#
# Throughput tuning: lightning.qubit scales sublinearly with threads, so
# several jobs with few threads each usually beat one job with all threads.
# Benchmark first:  OMP_NUM_THREADS=4 python src/qgan/benchmark_compute.py --versions v2 --reps 2
# then set PARALLEL_JOBS x THREADS_PER_JOB ~= physical cores.
#
# Usage:  PARALLEL_JOBS=4 THREADS_PER_JOB=4 bash scripts/run_grid.sh
# Resume: completed jobs (model_manifest.json present) are skipped.

set -u
cd "$(dirname "$0")/.."

PARALLEL_JOBS=${PARALLEL_JOBS:-4}
THREADS_PER_JOB=${THREADS_PER_JOB:-4}
EPOCHS=${EPOCHS:-50}
mkdir -p logs outputs/ablation

JOBLIST=$(mktemp)
for cls in 4 9 8; do
  for cfg in "v1 log_minmax" "v2 log_minmax" "v2 quantile"; do
    for seed in 42 43; do
      echo "$cls $cfg $seed" >> "$JOBLIST"
    done
  done
done

echo "$(wc -l < "$JOBLIST") jobs | $PARALLEL_JOBS parallel x $THREADS_PER_JOB threads | $EPOCHS epochs"

export THREADS_PER_JOB EPOCHS
xargs -P "$PARALLEL_JOBS" -I{} bash -c '
  set -- {}
  cls=$1; circ=$2; prep=$3; seed=$4
  tag="c${cls}_${circ}_${prep}_s${seed}"
  out="outputs/ablation/${tag}"
  if [ -f "${out}/model_manifest.json" ]; then
    echo "[skip] ${tag} already complete"; exit 0
  fi
  echo "[start] ${tag}"
  OMP_NUM_THREADS=$THREADS_PER_JOB python src/qgan/train.py "$cls" \
      --circuit "$circ" --preproc "$prep" --seed "$seed" \
      --epochs "$EPOCHS" --out-dir "$out" > "logs/${tag}.log" 2>&1 \
    && echo "[done ] ${tag}" || echo "[FAIL ] ${tag} (see logs/${tag}.log)"
' < "$JOBLIST"
rm -f "$JOBLIST"

echo "grid finished; summarize with: python scripts/summarize_runs.py outputs/ablation"
