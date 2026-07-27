"""Per-class preprocessing shared by train.py and generate.py.

Two options, selectable per training run and recorded in the model manifest
so generation can never apply the wrong inverse:

  log_minmax  - FROZEN original: log1p -> MinMax to [0, pi]. Kept for
                ablation against impg@735bdfc behavior.
  quantile    - per-feature QuantileTransformer to uniform, scaled to
                [0, pi]. A smooth generator only has to produce a roughly
                uniform marginal; the inverse maps through the empirical
                quantiles, so heavy tails and multi-modality of the MARGINALS
                are recovered by construction.

IMPORTANT evaluation consequence of `quantile`: per-feature marginal metrics
(JS, per-feature Wasserstein) become nearly meaningless as quality evidence,
because marginals match by construction. Judge such runs on JOINT structure:
correlation-matrix distance, C2ST, downstream utility.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

PREPROC_KINDS = ("log_minmax", "quantile")


def fit_preproc(X_raw: np.ndarray, kind: str, seed: int = 42):
    """Fit on raw-scale data; return (state, X_transformed in [0, pi])."""
    if kind == "log_minmax":
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X = scaler.fit_transform(np.log1p(X_raw))
    elif kind == "quantile":
        scaler = QuantileTransformer(
            n_quantiles=min(1000, len(X_raw)),
            output_distribution="uniform",
            subsample=1_000_000_000,
            random_state=seed,
        )
        X = scaler.fit_transform(X_raw) * np.pi
    else:
        raise ValueError(f"unknown preproc kind {kind!r}; choose from {PREPROC_KINDS}")
    return {"kind": kind, "scaler": scaler}, X


def inverse_preproc(state: dict, X_pi: np.ndarray) -> np.ndarray:
    """Map generator outputs in [0, pi] back to the original feature scale."""
    kind, scaler = state["kind"], state["scaler"]
    if kind == "log_minmax":
        return np.expm1(scaler.inverse_transform(X_pi))
    if kind == "quantile":
        return scaler.inverse_transform(np.clip(X_pi / np.pi, 0.0, 1.0))
    raise ValueError(f"unknown preproc kind {kind!r}")
