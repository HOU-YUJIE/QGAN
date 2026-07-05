# Running QGAN jobs on Kaggle

One Kaggle notebook = one atomic job (class, circuit, preproc, seed).
Jobs cannot resume mid-training; keep each under the 12 h session limit
(re-benchmark on Kaggle first - its 4-core CPU is likely 1.5-3x slower
per step than a 16-core desktop).

## One-time setup
1. Zip the repo WITHOUT bulky outputs (`src/`, `scripts/`,
   `data/processed/features16_train.csv`, `requirements.txt`) and upload it
   as a private Kaggle Dataset, e.g. `qgan-repo`.
2. Notebook settings: **Accelerator = None (CPU)** - lightning.qubit is a
   CPU simulator; a GPU adds queue time and nothing else. Internet = On
   (for pip). Persistence not required.

## Notebook cells

```python
# Cell 1 - environment (pin versions to requirements.txt)
!pip install -q pennylane==0.45.1 pennylane-lightning==0.45.0
!cp -r /kaggle/input/qgan-repo/* /kaggle/working/
%cd /kaggle/working

# Cell 2 - benchmark THIS machine once per account, then delete the cell
!OMP_NUM_THREADS=4 python src/qgan/benchmark_compute.py --versions v2 --reps 2 --epochs 50

# Cell 3 - the job (edit the four parameters per notebook copy)
CLS, CIRC, PREP, SEED, EPOCHS = 8, "v2", "quantile", 42, 50
tag = f"c{CLS}_{CIRC}_{PREP}_s{SEED}"
!OMP_NUM_THREADS=4 python src/qgan/train.py {CLS} --circuit {CIRC} \
    --preproc {PREP} --seed {SEED} --epochs {EPOCHS} \
    --out-dir outputs/ablation/{tag}

# Cell 4 - package results for download
!cd outputs && zip -r /kaggle/working/{tag}.zip ablation/{tag}
```

Duplicate the notebook per job (Kaggle allows several concurrent CPU
sessions - check your account's current quota in the UI). Download each
`*.zip`, unzip into the local `outputs/ablation/`, then:

```bash
python scripts/summarize_runs.py outputs/ablation
```

## Job placement suggestion
Put the heaviest jobs (class 8: ~2x the cost of class 4) on Kaggle first
and run the light ones locally in parallel; `run_grid.sh` skips any job
whose manifest already exists, so local and Kaggle work can be merged
freely without duplication.
