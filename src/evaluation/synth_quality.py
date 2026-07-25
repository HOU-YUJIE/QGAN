"""Synthetic-data quality evaluation: PURE synthetic vs real, per class.

Replaces src/qgan/evaluate.py and experiments/compare_js_distance.py. The
old JS comparison scored real+synthetic MIXTURES against real data, which
rewarded whichever condition contained more real rows; and it used 50
equal-width global bins, which collapses heavy-tailed features into one bin
(JS artificially ~0). Both errors are fixed here.

Four complementary metric families per (generator, class):

  marginals   js_quantile  - per-feature JS distance on 20 quantile bins
                             derived from the REAL data (discrete features
                             use their value support); mean over features
              std_w1       - per-feature Wasserstein-1 / real std; mean
  joint       corr_mad     - mean |corr(syn) - corr(real)| over pairs
              corr_frob    - Frobenius norm of the same difference
  detection   c2st_auc     - 5-fold CV RandomForest AUC distinguishing
                             real vs synthetic (0.5 = indistinguishable,
                             1.0 = trivially separable)
  memorization dcr_ratio   - median NN distance (synthetic -> real train) /
                             median leave-one-out NN distance (real -> real),
                             standardized features. << 1 means copying.
              copy_frac    - fraction of synthetic rows closer to a real row
                             than the 1st percentile of real-real distances.

READING GUIDE for quantile-preprocessed generators: marginal metrics are
near-perfect BY CONSTRUCTION - quality claims must rest on corr_*, c2st_auc
and downstream utility. A great-looking table with dcr_ratio << 1 is not a
good generator; it is a copying machine.

Reference distribution is the class's REAL TRAINING rows (what the
generator modeled); memorization is only defined against training rows.

Usage:
    python src/evaluation/synth_quality.py                       # all generators, minority classes
    python src/evaluation/synth_quality.py --generators qgan --classes 4 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import (
    DISCRETE_FEATURES, FEATURES16_TRAIN, LABEL_COLUMN, MANUAL_FEATURES_16,
    RESULTS_DIR, SEED, get_majority_labels, synthetic_file,
)

GENERATORS = ["qgan", "ctgan", "classical"]
N_QUANTILE_BINS = 20


def js_quantile(real: np.ndarray, syn: np.ndarray, discrete: bool) -> float:
    if discrete:
        support = np.unique(real)
        edges = np.concatenate([support - 0.5, [support.max() + 0.5]])
    else:
        edges = np.unique(np.quantile(real, np.linspace(0, 1, N_QUANTILE_BINS + 1)))
        if len(edges) < 3:  # (near-)constant feature
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5]) if len(edges) == 1 else edges
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
    p, _ = np.histogram(real, bins=edges)
    q, _ = np.histogram(syn, bins=edges)
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    return float(jensenshannon(p, q, base=2))


def c2st_auc(real: np.ndarray, syn: np.ndarray, seed: int) -> float:
    rng = np.random.RandomState(seed)
    n = min(len(real), len(syn))
    Xr = real[rng.choice(len(real), n, replace=False)]
    Xs = syn[rng.choice(len(syn), n, replace=False)]
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    proba = cross_val_predict(clf, X, y, method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    return float(roc_auc_score(y, proba))


def memorization(real: np.ndarray, syn: np.ndarray) -> tuple:
    sc = StandardScaler().fit(real)
    R, S = sc.transform(real), sc.transform(syn)
    nn = NearestNeighbors(n_neighbors=2).fit(R)
    real_d = nn.kneighbors(R)[0][:, 1]          # leave-one-out: 2nd neighbor
    syn_d = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(S)[0][:, 0]
    dcr_ratio = float(np.median(syn_d) / max(np.median(real_d), 1e-12))
    thresh = np.percentile(real_d, 1)
    copy_frac = float((syn_d <= thresh + 1e-12).mean())
    return dcr_ratio, copy_frac


def evaluate_one(generator: str, label: int, real_df: pd.DataFrame, seed: int) -> dict:
    path = synthetic_file(generator, label)
    if not os.path.exists(path):
        return {"generator": generator, "label": label, "status": "missing"}
    syn_df = pd.read_csv(path)
    real = real_df[MANUAL_FEATURES_16].values.astype(float)
    syn = syn_df[MANUAL_FEATURES_16].values.astype(float)

    js = np.mean([js_quantile(real[:, j], syn[:, j],
                              MANUAL_FEATURES_16[j] in DISCRETE_FEATURES)
                  for j in range(real.shape[1])])
    stds = real.std(axis=0) + 1e-9
    w1 = np.mean([wasserstein_distance(real[:, j], syn[:, j]) / stds[j]
                  for j in range(real.shape[1])])

    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.nan_to_num(np.corrcoef(real, rowvar=False))
        cs = np.nan_to_num(np.corrcoef(syn, rowvar=False))
    diff = np.abs(cs - cr)
    iu = np.triu_indices_from(diff, k=1)

    dcr_ratio, copy_frac = memorization(real, syn)

    return {"generator": generator, "label": int(label), "status": "ok",
            "n_real": len(real), "n_syn": len(syn),
            "js_quantile": round(float(js), 4), "std_w1": round(float(w1), 4),
            "corr_mad": round(float(diff[iu].mean()), 4),
            "corr_frob": round(float(np.linalg.norm(cs - cr)), 4),
            "c2st_auc": round(c2st_auc(real, syn, seed), 4),
            "dcr_ratio": round(dcr_ratio, 4), "copy_frac": round(copy_frac, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generators", nargs="+", default=GENERATORS)
    p.add_argument("--classes", nargs="+", type=int, default=None,
                   help="default: all minority classes")
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()

    df = pd.read_csv(FEATURES16_TRAIN)
    if a.classes is None:
        majority = get_majority_labels(FEATURES16_TRAIN)
        classes = sorted(set(df[LABEL_COLUMN].unique()) - majority)
    else:
        classes = a.classes
    print(f"generators: {a.generators} | classes: {classes}")

    rows = []
    for gen in a.generators:
        for label in classes:
            r = evaluate_one(gen, label, df[df[LABEL_COLUMN] == label], a.seed)
            rows.append(r)
            if r["status"] == "ok":
                print(f"[{gen:9s} class {label}] JS {r['js_quantile']:.3f} | "
                      f"W1 {r['std_w1']:.3f} | corrMAD {r['corr_mad']:.3f} | "
                      f"C2ST {r['c2st_auc']:.3f} | DCR {r['dcr_ratio']:.3f} "
                      f"copy% {r['copy_frac']:.1%}")
            else:
                print(f"[{gen:9s} class {label}] synthetic file missing - skipped")

    res = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "synth_quality.csv"), index=False)

    if len(res):
        metrics = ["js_quantile", "std_w1", "corr_mad", "c2st_auc", "dcr_ratio", "copy_frac"]
        summary = res.groupby("generator")[metrics].mean().round(4)
        text = ("\nmean over classes (lower better except c2st_auc->0.5, dcr_ratio->~1):\n"
                + summary.to_string()
                + "\n\nreading guide: for quantile-preprocessed runs, judge on corr_* and"
                + "\nc2st_auc; js/w1 match by construction. dcr_ratio << 1 or copy_frac"
                + "\nhigh = memorization, disqualifying regardless of other scores; c2st_auc"
                + "\nwell BELOW 0.5 is also a memorization signature (copies pass as real).")
        with open(os.path.join(RESULTS_DIR, "synth_quality.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
