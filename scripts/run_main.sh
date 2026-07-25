#!/usr/bin/env bash
# Main experiment orchestration (config: circuit v2, preproc quantile,
# 100 epochs, patience 6 - all defaults in train.py/config.py).
#
# Seed protocol (decided from the screening grid):
#   classes 2, 4  (<600 training rows, high GAN-seed variance):
#       3 seeds each, best selected by validation best_score -> canonical dir
#   classes 1, 5, 7, 8, 9: single seed (42)
# The same protocol applies to the classical control for fairness.
#
# Resume-friendly: any stage whose output manifest already exists is skipped,
# so QGAN training can be done on Kaggle (drop result dirs into
# outputs/models/... first) and this script stitches the rest locally.
#
# Usage:
#   bash scripts/run_main.sh                # everything (QGAN training local)
#   SKIP_QGAN_TRAIN=1 bash scripts/run_main.sh   # QGAN dirs came from Kaggle

set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

MULTI_SEED_CLASSES="2 4"
SINGLE_SEED_CLASSES="1 5 7 8 9"
MULTI_SEEDS="42 43 44"
SKIP_QGAN_TRAIN=${SKIP_QGAN_TRAIN:-0}
mkdir -p logs

canon () {  # canonical model dir for generator $1 class $2
  python -c "import sys; sys.path.insert(0,'.'); from src.config import generator_model_dir; print(generator_model_dir('$1', $2))"
}

train_job () {  # $1=circuit(v2|classical) $2=class $3=seed $4=outdir
  if [ -f "$4/model_manifest.json" ]; then echo "[skip] $1 c$2 s$3"; return 0; fi
  echo "[train] $1 c$2 s$3 -> $4"
  python -u src/qgan/train.py "$2" --circuit "$1" --seed "$3" --out-dir "$4" \
      > "logs/main_$1_c$2_s$3.log" 2>&1 || echo "[FAIL] $1 c$2 s$3 (see log)"
}

select_best () {  # $1=circuit $2=class : copy best seedsel run into canonical dir
  python - "$1" "$2" <<'PY'
import glob, json, os, shutil, sys
sys.path.insert(0, ".")
from src.config import generator_model_dir
gen, label = sys.argv[1], int(sys.argv[2])
kind = "qgan" if gen != "classical" else "classical"
cands = []
for m in glob.glob(f"outputs/models/seedsel/{kind}/{label}/s*/model_manifest.json"):
    with open(m) as f:
        cands.append((json.load(f)["best_score"], os.path.dirname(m)))
if not cands:
    sys.exit(f"no seedsel runs for {kind} class {label}")
score, src = min(cands)
dst = generator_model_dir(kind, label)
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), dst)
with open(os.path.join(dst, "seed_selection.json"), "w") as f:
    json.dump({"selected_dir": src, "best_score": score,
               "candidates": sorted([(round(s, 4), os.path.basename(d)) for s, d in cands])}, f, indent=2)
print(f"[select] {kind} c{label}: {os.path.basename(src)} (score {score:.4f}) -> {dst}")
PY
}

# ---- Phase 1: QGAN training -------------------------------------------------
if [ "$SKIP_QGAN_TRAIN" != "1" ]; then
  for c in $SINGLE_SEED_CLASSES; do train_job v2 "$c" 42 "$(canon qgan "$c")"; done
  for c in $MULTI_SEED_CLASSES; do
    for s in $MULTI_SEEDS; do train_job v2 "$c" "$s" "outputs/models/seedsel/qgan/$c/s$s"; done
  done
fi
for c in $MULTI_SEED_CLASSES; do
  [ -f "$(canon qgan "$c")/seed_selection.json" ] || select_best v2 "$c"
done

# ---- Phase 2: classical control (fast, always local) ------------------------
for c in $SINGLE_SEED_CLASSES; do train_job classical "$c" 42 "$(canon classical "$c")"; done
for c in $MULTI_SEED_CLASSES; do
  for s in $MULTI_SEEDS; do train_job classical "$c" "$s" "outputs/models/seedsel/classical/$c/s$s"; done
  [ -f "$(canon classical "$c")/seed_selection.json" ] || select_best classical "$c"
done

# ---- Phase 3: generation ----------------------------------------------------
for c in $MULTI_SEED_CLASSES $SINGLE_SEED_CLASSES; do
  python -u src/qgan/generate.py "$c" --generator qgan      >> logs/main_generate.log 2>&1
  python -u src/qgan/generate.py "$c" --generator classical >> logs/main_generate.log 2>&1
done
echo "[done] generation (see logs/main_generate.log)"

# ---- Phase 4: CTGAN baseline ------------------------------------------------
python -u src/baselines/train_ctgan.py --all > logs/main_ctgan.log 2>&1
echo "[done] ctgan"

# ---- Phase 5-7: datasets, MLP, quality --------------------------------------
python -u src/fusion/build_datasets.py
python -u src/mlp/train.py --seeds 10
python -u src/evaluation/synth_quality.py
echo "[ALL DONE] see outputs/results/"
