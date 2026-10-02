"""
paper_all_contributions_wifi_rtt_rss.py
=======================================

Comprehensive paper experiment for hybrid WiFi RTT-RSS indoor positioning.

This script consolidates the contributions that survived our earlier ablation
experiments and that can plausibly participate in the final paper:

CONFIRMED / RETAINED COMPONENTS
-------------------------------
1. Hybrid WiFi RTT + RSS fingerprints.
2. Random-subspace Random Forest:
       baseline : max_features = 1.0
       proposed : max_features = 0.7
3. Reference-point (RP) grouping:
       all scans from one physical (X,Y) reference point stay in the same fold.
4. Full-data final localization model:
       the final proposed point predictor is trained on 100% of official TRAIN.
5. Grouped out-of-fold (OOF) residuals:
       used to estimate residual geometry without scan leakage.
6. Covariance-aware anisotropic uncertainty:
       Mahalanobis nonconformity score -> elliptical region.
7. Original-style uncertainty baselines:
       axis-aligned rectangle and Euclidean circle.
8. Paper-compatible localization metric:
       mean of the X- and Y-coordinate RMSEs, multiplied by physical grid size.
       We ALSO report Euclidean mean/RMSE and percentiles so no metric is hidden.

IMPORTANT SCIENTIFIC DISTINCTION
--------------------------------
Two uncertainty protocols are reported:

A) STRICT GROUPED SPLIT-CONFORMAL
   - RPs are split into fit and calibration groups.
   - The ellipse covariance shape is learned ONLY from grouped OOF residuals
     inside the fit set.
   - The held-out calibration set is used ONLY to obtain the final conformal
     quantile.
   - This is the clean split-conformal comparison.

B) FULL-TRAIN GROUPED-OOF EMPIRICAL CALIBRATION
   - Grouped OOF residuals are obtained from all official TRAIN RPs.
   - A final subspace RF is retrained on 100% of official TRAIN.
   - OOF residuals determine circle/ellipse empirical quantiles.
   - This protocol is data-efficient and is useful experimentally, BUT the
     residuals come from multiple fold-specific predictors. Therefore the
     script labels it "OOF empirical" and DOES NOT claim ordinary
     split-conformal finite-sample validity.

This distinction is intentional and should remain in the paper.

EXPECTED DATA DIRECTORY
-----------------------
dataset/
    database_building_train.csv
    database_building_test.csv
    database_office_train.csv
    database_office_test.csv
    database_apartment_train.csv
    database_apartment_test.csv

Missing-value sentinels used by the source dataset/paper:
    RTT = 100000
    RSS = -200

PHYSICAL GRID SIZES
-------------------
Building  : 0.600 m
Office    : 0.455 m
Apartment : 0.480 m

PUBLISHED HYBRID RTT+RSS REFERENCE VALUES
-----------------------------------------
Building  : 0.60 m
Office    : 0.38 m
Apartment : 0.59 m

These are included only as a reference line/table.  The script recomputes the
paper-compatible metric directly from predictions so that comparisons are
apples-to-apples.

OUTPUT
------
paper_results_all_contributions/
    tables/*.csv
    tables/*.tex
    plots/*.png
    plots/*.pdf

Main CSV tables:
    dataset_summary.csv
    point_localization_results.csv
    published_comparison.csv
    grouped_cv_fold_results.csv
    grouped_cv_summary.csv
    split_conformal_results.csv
    full_oof_uncertainty_results.csv
    oof_residual_summary.csv
    feature_importance.csv
    cluster_bootstrap_comparison.csv
    contribution_summary.csv

PLOTS
-----
Dataset:
    workflow
    train/test RP maps
    samples-per-RP histograms
    AP missingness / visibility
    RTT/RSS distributions
    feature-correlation heatmaps

Localization:
    paper-metric comparison
    CDFs of physical Euclidean error
    error boxplots
    percentile profiles
    grouped-CV fold plots
    test spatial error maps
    feature importance
    compute comparison

Uncertainty:
    OOF residual scatter
    residual covariance ellipse
    OOF residual CDF
    nominal-vs-empirical coverage
    coverage-vs-area
    circle-vs-ellipse areas
    example prediction regions

RUN
---
    python paper_all_contributions_wifi_rtt_rss.py

Dependencies:
    numpy
    pandas
    matplotlib
    scikit-learn

No seaborn is required.
"""

from __future__ import annotations

import gc
import json
import math
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
RNG = np.random.default_rng(SEED)

DATASET_DIR = Path("dataset")
OUT_DIR = Path("paper_results_all_contributions")
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RTT_MISSING = 100000.0
RSS_MISSING = -200.0

# Core RF settings retained from the experiments.
N_TREES = 300
BASELINE_MAX_FEATURES = 1.0
PROPOSED_MAX_FEATURES = 0.7
MIN_SAMPLES_LEAF = 1

# Group-aware experimental protocol.
CV_FOLDS = 5
OOF_FOLDS = 5
CALIBRATION_FRACTION = 0.25

# Conformal confidence levels.
CONFIDENCE_LEVELS = [0.90, 0.95]

# Timing is noisy; median of repeated predictions is more stable.
PREDICTION_REPEATS = 5

# Cluster bootstrap by physical RP for paired uncertainty on point-error change.
BOOTSTRAP_REPEATS = 1000

# How many test examples to draw in example-region plots.
N_REGION_EXAMPLES = 8

# Save publication plots as both PNG and vector PDF.
SAVE_PNG = True
SAVE_PDF = True
FIG_DPI = 300

# To reduce very large dataset plots only.  Models still use all rows.
MAX_PLOT_ROWS = 20000

# Source-paper values for hybrid RTT+RSS point localization.
PUBLISHED_HYBRID_METRIC_M = {
    "building": 0.60,
    "office": 0.38,
    "apartment": 0.59,
}

# Physical grid size of one coordinate unit.
GRID_SIZE_M = {
    "building": 0.600,
    "office": 0.455,
    "apartment": 0.480,
}

DATASETS = {
    "building": (
        "database_building_train.csv",
        "database_building_test.csv",
    ),
    "office": (
        "database_office_train.csv",
        "database_office_test.csv",
    ),
    "apartment": (
        "database_apartment_train.csv",
        "database_apartment_test.csv",
    ),
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def save_figure(fig, stem: str) -> None:
    """Save a figure in publication-friendly raster and vector formats."""
    fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(PLOT_DIR / f"{stem}.png", dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(PLOT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, filename: str, latex: bool = True) -> None:
    """Save a CSV and, when possible, a LaTeX table."""
    csv_path = TABLE_DIR / filename
    df.to_csv(csv_path, index=False)

    if latex:
        tex_path = csv_path.with_suffix(".tex")
        try:
            tex_path.write_text(
                df.to_latex(index=False, escape=True, float_format="%.4f"),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[warning] Could not export {tex_path.name}: {exc}")


def pct_change(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        return np.nan
    if abs(baseline) <= 1e-15:
        return np.nan
    return 100.0 * (candidate - baseline) / baseline


def finite_quantile_higher(values: np.ndarray, q: float) -> float:
    """Compatibility helper for NumPy versions with/without method=."""
    values = np.asarray(values, dtype=float)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def conformal_quantile(scores: np.ndarray, confidence: float) -> float:
    """
    Standard finite-sample split-conformal quantile:
        ceil((n+1)*(1-alpha)) / n
    using a higher empirical quantile.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        raise ValueError("No finite calibration scores.")

    alpha = 1.0 - confidence
    level = math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores)
    level = min(1.0, max(0.0, level))
    return finite_quantile_higher(scores, level)


def conformal_quantile_rp_corrected(
    scores: np.ndarray,
    confidence: float,
    n_groups: int,
) -> float:
    """
    Split-conformal quantile when calibration scans are clustered by RP.

    The ~120 scans of one RP are strongly correlated, so the effective number
    of exchangeable calibration units is the number of calibration RPs, not
    the number of scans.  Scores are pooled, but the finite-sample correction
    uses n = n_groups:
        level = ceil((G+1)(1-alpha)) / G
    This removes the under-coverage of the scan-level rule when few RPs are
    held out for calibration, at the cost of larger regions.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    level = min(1.0, math.ceil((n_groups + 1) * confidence) / n_groups)
    return finite_quantile_higher(scores, level)


def sample_rows_for_plot(df: pd.DataFrame, max_rows: int = MAX_PLOT_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=SEED)


# =============================================================================
# DATA LOADING / FEATURE EXTRACTION
# =============================================================================

def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    return df.drop(columns=cols) if cols else df


def load_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_name, test_name = DATASETS[dataset]

    train_path = DATASET_DIR / train_name
    test_path = DATASET_DIR / test_name

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    train_df = drop_unnamed(pd.read_csv(train_path, low_memory=False))
    test_df = drop_unnamed(pd.read_csv(test_path, low_memory=False))

    required = {"X", "Y"}
    if not required.issubset(train_df.columns):
        raise ValueError(f"{dataset}: TRAIN is missing X/Y columns.")
    if not required.issubset(test_df.columns):
        raise ValueError(f"{dataset}: TEST is missing X/Y columns.")

    return train_df, test_df


def ap_number(col: str) -> int:
    digits = "".join(ch for ch in str(col) if ch.isdigit())
    return int(digits) if digits else 999999


def detect_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Detect RTT and RSS columns only.
    Columns such as 'LoS APs' are intentionally excluded.
    """
    rtt_cols = [c for c in df.columns if "RTT" in str(c).upper()]
    rss_cols = [c for c in df.columns if "RSS" in str(c).upper()]

    rtt_cols = sorted(rtt_cols, key=ap_number)
    rss_cols = sorted(rss_cols, key=ap_number)

    if not rtt_cols:
        raise ValueError("No RTT columns detected.")
    if not rss_cols:
        raise ValueError("No RSS columns detected.")
    if len(rtt_cols) != len(rss_cols):
        raise ValueError(
            f"RTT/RSS AP mismatch: {len(rtt_cols)} RTT columns vs "
            f"{len(rss_cols)} RSS columns."
        )

    return rtt_cols, rss_cols


def extract_features_target(
    df: pd.DataFrame,
    rtt_cols: list[str],
    rss_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Keep source-paper sentinel encoding:
        unheard RTT -> 100000
        unheard RSS -> -200

    RF can exploit the "not heard" state directly.
    """
    RTT = (
        df[rtt_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    RSS = (
        df[rss_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    y = (
        df[["X", "Y"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    RTT = np.where(np.isnan(RTT), RTT_MISSING, RTT)
    RSS = np.where(np.isnan(RSS), RSS_MISSING, RSS)

    # Ordering does not alter RF semantics, but explicit names are saved.
    X = np.hstack([RTT, RSS]).astype(np.float64, copy=False)
    feature_names = list(rtt_cols) + list(rss_cols)

    return X, y, feature_names


def make_rp_groups(df: pd.DataFrame) -> np.ndarray:
    """Every scan from the same physical (X,Y) location receives one group ID."""
    coords = pd.MultiIndex.from_frame(df[["X", "Y"]].astype(float))
    labels, _ = pd.factorize(coords)
    return np.asarray(labels, dtype=int)


def coordinate_set(df: pd.DataFrame) -> set[tuple[float, float]]:
    return set(
        map(tuple, df[["X", "Y"]].astype(float).to_numpy())
    )


# =============================================================================
# RF MODEL
# =============================================================================

def build_rf(max_features: float, seed: int = SEED) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_TREES,
        max_depth=None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=float(max_features),
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )


def timed_fit(model, X: np.ndarray, y: np.ndarray) -> float:
    gc.collect()
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def timed_predict(
    model,
    X: np.ndarray,
    repeats: int = PREDICTION_REPEATS,
) -> tuple[np.ndarray, float]:
    # Warm-up to avoid counting one-time worker initialization.
    _ = model.predict(X[: min(32, len(X))])

    times = []
    pred = None
    for _ in range(repeats):
        start = time.perf_counter()
        pred = model.predict(X)
        times.append(time.perf_counter() - start)

    return pred, float(np.median(times))


def model_size_mb(model) -> float:
    try:
        blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        return len(blob) / (1024.0 * 1024.0)
    except Exception:
        return np.nan


# =============================================================================
# LOCALIZATION METRICS
# =============================================================================

def physical_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_size_m: float,
) -> np.ndarray:
    """Return [dx,dy] residuals in metres."""
    return (y_true - y_pred) * float(grid_size_m)


def localization_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_size_m: float,
) -> dict[str, float]:
    """
    Report both the exact paper-compatible coordinate-RMSE metric and
    physically intuitive Euclidean metrics.

    paper_metric_m:
        mean( RMSE_x, RMSE_y ) * grid_size

    This matches:
        mean_squared_error(y_true, y_pred, squared=False)
    with sklearn's default multioutput='uniform_average', then converted
    from grid units to metres.
    """
    residual_grid = y_true - y_pred
    residual_m = residual_grid * grid_size_m

    coord_rmse_grid = np.sqrt(np.mean(residual_grid ** 2, axis=0))
    coord_rmse_m = coord_rmse_grid * grid_size_m

    paper_metric_m = float(np.mean(coord_rmse_m))

    radial_m = np.linalg.norm(residual_m, axis=1)

    return {
        "paper_metric_m": paper_metric_m,
        "rmse_x_m": float(coord_rmse_m[0]),
        "rmse_y_m": float(coord_rmse_m[1]),
        "mean_euclidean_m": float(np.mean(radial_m)),
        "median_euclidean_m": float(np.median(radial_m)),
        "rmse_euclidean_m": float(np.sqrt(np.mean(radial_m ** 2))),
        "p50_euclidean_m": float(np.quantile(radial_m, 0.50)),
        "p80_euclidean_m": float(np.quantile(radial_m, 0.80)),
        "p90_euclidean_m": float(np.quantile(radial_m, 0.90)),
        "p95_euclidean_m": float(np.quantile(radial_m, 0.95)),
        "max_euclidean_m": float(np.max(radial_m)),
    }


# =============================================================================
# POINT-LOCALIZATION EXPERIMENT
# =============================================================================

def evaluate_full_train_model(
    dataset: str,
    model_name: str,
    max_features: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    grid_size_m: float,
) -> tuple[dict, np.ndarray, RandomForestRegressor]:
    model = build_rf(max_features=max_features)

    train_time = timed_fit(model, X_train, y_train)
    pred, predict_time = timed_predict(model, X_test)
    metrics = localization_metrics(y_test, pred, grid_size_m)

    row = {
        "dataset": dataset,
        "model": model_name,
        "n_trees": N_TREES,
        "max_features": max_features,
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "predict_us_per_sample": predict_time / max(len(y_test), 1) * 1e6,
        "model_size_mb": model_size_mb(model),
        **metrics,
    }

    return row, pred, model


# =============================================================================
# GROUPED CV
# =============================================================================

def grouped_cv_compare(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    grid_size_m: float,
) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    n_splits = min(CV_FOLDS, len(unique_groups))
    if n_splits < 2:
        raise ValueError(f"{dataset}: insufficient RPs for GroupKFold.")

    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(X), 1))
    folds = list(splitter.split(dummy, y, groups))

    rows = []

    configs = [
        ("RF300-F1.0", BASELINE_MAX_FEATURES),
        ("RF300-F0.7", PROPOSED_MAX_FEATURES),
    ]

    for model_name, max_features in configs:
        pooled_pred = np.full_like(y, np.nan, dtype=float)

        for fold_id, (tr, va) in enumerate(folds, start=1):
            model = build_rf(max_features=max_features)

            train_time = timed_fit(model, X[tr], y[tr])
            pred, predict_time = timed_predict(model, X[va])
            pooled_pred[va] = pred
            m = localization_metrics(y[va], pred, grid_size_m)

            rows.append({
                "dataset": dataset,
                "model": model_name,
                "fold": fold_id,
                "train_rps": len(np.unique(groups[tr])),
                "validation_rps": len(np.unique(groups[va])),
                "train_time_s": train_time,
                "predict_time_s": predict_time,
                **m,
            })

            del model
            gc.collect()

        # Metrics on the pooled out-of-fold predictions.  Averaging per-fold
        # RMSEs is not an RMSE over the dataset; the pooled row is the one
        # reported, the per-fold rows provide the spread.
        rows.append({
            "dataset": dataset,
            "model": model_name,
            "fold": "pooled",
            "train_rps": np.nan,
            "validation_rps": len(np.unique(groups)),
            "train_time_s": np.nan,
            "predict_time_s": np.nan,
            **localization_metrics(y, pooled_pred, grid_size_m),
        })

    return pd.DataFrame(rows)


# =============================================================================
# GROUPED OOF PREDICTIONS
# =============================================================================

def grouped_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_features: float,
    n_splits: int = OOF_FOLDS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Produce one out-of-fold prediction for every row, with entire RPs held out.
    """
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))

    if n_splits < 2:
        raise ValueError("Not enough RP groups for OOF prediction.")

    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(X), 1))

    oof_pred = np.full_like(y, np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows = []

    for fold_id, (tr, va) in enumerate(
        splitter.split(dummy, y, groups),
        start=1,
    ):
        model = build_rf(max_features=max_features)
        train_time = timed_fit(model, X[tr], y[tr])
        pred, pred_time = timed_predict(model, X[va])

        oof_pred[va] = pred
        fold_ids[va] = fold_id

        fold_rows.append({
            "fold": fold_id,
            "train_rows": len(tr),
            "validation_rows": len(va),
            "train_rps": len(np.unique(groups[tr])),
            "validation_rps": len(np.unique(groups[va])),
            "train_time_s": train_time,
            "predict_time_s": pred_time,
        })

        del model
        gc.collect()

    if np.isnan(oof_pred).any():
        raise RuntimeError("Incomplete OOF predictions.")

    return oof_pred, fold_ids, pd.DataFrame(fold_rows)


# =============================================================================
# COVARIANCE / ELLIPSE
# =============================================================================

def regularized_covariance(residuals_m: np.ndarray) -> np.ndarray:
    """
    Estimate a 2x2 residual covariance and add only a tiny numerical ridge.
    No tuned shrinkage parameter is used.
    """
    residuals_m = np.asarray(residuals_m, dtype=float)

    if residuals_m.ndim != 2 or residuals_m.shape[1] != 2:
        raise ValueError("Residuals must have shape (n,2).")

    Sigma = np.cov(residuals_m, rowvar=False, ddof=1)

    if Sigma.shape != (2, 2) or not np.all(np.isfinite(Sigma)):
        scale = float(np.nanmean(np.var(residuals_m, axis=0)))
        scale = max(scale, 1e-6)
        Sigma = np.eye(2) * scale

    scale = float(np.trace(Sigma) / 2.0)
    ridge = max(1e-10, 1e-6 * max(scale, 1e-10))
    Sigma = Sigma + ridge * np.eye(2)

    # Ensure positive definiteness.
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    floor = max(1e-10, 1e-8 * max(float(np.max(eigvals)), 1.0))
    eigvals = np.maximum(eigvals, floor)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return Sigma


def mahalanobis_scores(residuals_m: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(Sigma)
    values = np.einsum(
        "ni,ij,nj->n",
        residuals_m,
        inv,
        residuals_m,
    )
    values = np.maximum(values, 0.0)
    return np.sqrt(values)


def ellipse_area(q: float, Sigma: np.ndarray) -> float:
    return float(
        math.pi
        * q ** 2
        * math.sqrt(max(float(np.linalg.det(Sigma)), 0.0))
    )


def ellipse_axes_angle(q: float, Sigma: np.ndarray) -> tuple[float, float, float]:
    """
    Return full ellipse width, full height, rotation angle in degrees.
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width = 2.0 * q * math.sqrt(float(eigvals[0]))
    height = 2.0 * q * math.sqrt(float(eigvals[1]))

    major_vec = eigvecs[:, 0]
    angle = math.degrees(math.atan2(major_vec[1], major_vec[0]))

    return width, height, angle


# =============================================================================
# STRICT GROUPED SPLIT-CONFORMAL
# =============================================================================

def grouped_fit_calibration_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split by physical RP, never by rows.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_FRACTION,
        random_state=SEED,
    )

    dummy = np.zeros((len(X), 1))
    fit_idx, cal_idx = next(splitter.split(dummy, y, groups))

    return fit_idx, cal_idx


def strict_split_conformal_experiment(
    dataset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    grid_size_m: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Compare original-style uncertainty and retained proposed components.

    Methods:
      1. Baseline RF F1.0 rectangle
      2. Baseline RF F1.0 circle
      3. Subspace RF F0.7 circle
      4. Subspace RF F0.7 OOF-covariance ellipse

    For the ellipse, Sigma is learned ONLY from grouped OOF residuals of the FIT
    set.  Calibration rows are then used only for the final q quantile.
    """
    fit_idx, cal_idx = grouped_fit_calibration_split(
        X_train,
        y_train,
        groups_train,
    )

    X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]
    groups_fit = groups_train[fit_idx]

    rows = []
    artifacts = {
        "fit_idx": fit_idx,
        "cal_idx": cal_idx,
    }

    # -------------------------------------------------------------------------
    # Baseline full-feature RF
    # -------------------------------------------------------------------------
    baseline = build_rf(BASELINE_MAX_FEATURES)
    baseline.fit(X_fit, y_fit)

    pred_cal_base = baseline.predict(X_cal)
    pred_test_base = baseline.predict(X_test)

    cal_resid_base = physical_residuals(y_cal, pred_cal_base, grid_size_m)
    test_resid_base = physical_residuals(y_test, pred_test_base, grid_size_m)

    cal_radial_base = np.linalg.norm(cal_resid_base, axis=1)
    test_radial_base = np.linalg.norm(test_resid_base, axis=1)

    for confidence in CONFIDENCE_LEVELS:
        # Rectangle baseline.
        qx = conformal_quantile(np.abs(cal_resid_base[:, 0]), confidence)
        qy = conformal_quantile(np.abs(cal_resid_base[:, 1]), confidence)

        rect_covered = (
            (np.abs(test_resid_base[:, 0]) <= qx)
            &
            (np.abs(test_resid_base[:, 1]) <= qy)
        )

        rows.append({
            "dataset": dataset,
            "protocol": "strict_grouped_split",
            "predictor": "RF300-F1.0",
            "region": "rectangle",
            "confidence": confidence,
            "empirical_coverage": float(np.mean(rect_covered)),
            "area_m2": float(4.0 * qx * qy),
            "q": np.nan,
            "qx_m": qx,
            "qy_m": qy,
            "fit_rows": len(fit_idx),
            "calibration_rows": len(cal_idx),
            "fit_rps": len(np.unique(groups_train[fit_idx])),
            "calibration_rps": len(np.unique(groups_train[cal_idx])),
        })

        # Circle baseline.
        q_circle = conformal_quantile(cal_radial_base, confidence)
        circle_covered = test_radial_base <= q_circle

        rows.append({
            "dataset": dataset,
            "protocol": "strict_grouped_split",
            "predictor": "RF300-F1.0",
            "region": "circle",
            "confidence": confidence,
            "empirical_coverage": float(np.mean(circle_covered)),
            "area_m2": float(math.pi * q_circle ** 2),
            "q": q_circle,
            "qx_m": np.nan,
            "qy_m": np.nan,
            "fit_rows": len(fit_idx),
            "calibration_rows": len(cal_idx),
            "fit_rps": len(np.unique(groups_train[fit_idx])),
            "calibration_rps": len(np.unique(groups_train[cal_idx])),
        })

    # -------------------------------------------------------------------------
    # Proposed subspace RF
    # -------------------------------------------------------------------------
    proposed = build_rf(PROPOSED_MAX_FEATURES)
    proposed.fit(X_fit, y_fit)

    pred_cal_prop = proposed.predict(X_cal)
    pred_test_prop = proposed.predict(X_test)

    cal_resid_prop = physical_residuals(y_cal, pred_cal_prop, grid_size_m)
    test_resid_prop = physical_residuals(y_test, pred_test_prop, grid_size_m)

    cal_radial_prop = np.linalg.norm(cal_resid_prop, axis=1)
    test_radial_prop = np.linalg.norm(test_resid_prop, axis=1)

    # Learn ellipse geometry from FIT data only via RP-grouped OOF prediction.
    oof_fit_pred, oof_fit_folds, oof_fit_timing = grouped_oof_predictions(
        X_fit,
        y_fit,
        groups_fit,
        max_features=PROPOSED_MAX_FEATURES,
        n_splits=OOF_FOLDS,
    )

    oof_fit_resid_m = physical_residuals(
        y_fit,
        oof_fit_pred,
        grid_size_m,
    )

    Sigma = regularized_covariance(oof_fit_resid_m)

    cal_maha = mahalanobis_scores(cal_resid_prop, Sigma)
    test_maha = mahalanobis_scores(test_resid_prop, Sigma)

    artifacts.update({
        "baseline_model": baseline,
        "proposed_model": proposed,
        "pred_test_baseline": pred_test_base,
        "pred_test_proposed": pred_test_prop,
        "cal_resid_proposed_m": cal_resid_prop,
        "test_resid_proposed_m": test_resid_prop,
        "oof_fit_resid_m": oof_fit_resid_m,
        "Sigma": Sigma,
    })

    for confidence in CONFIDENCE_LEVELS:
        # Same subspace predictor with a circle isolates base-model improvement.
        q_circle = conformal_quantile(cal_radial_prop, confidence)
        circle_covered = test_radial_prop <= q_circle

        rows.append({
            "dataset": dataset,
            "protocol": "strict_grouped_split",
            "predictor": "RF300-F0.7",
            "region": "circle",
            "confidence": confidence,
            "empirical_coverage": float(np.mean(circle_covered)),
            "area_m2": float(math.pi * q_circle ** 2),
            "q": q_circle,
            "qx_m": np.nan,
            "qy_m": np.nan,
            "fit_rows": len(fit_idx),
            "calibration_rows": len(cal_idx),
            "fit_rps": len(np.unique(groups_train[fit_idx])),
            "calibration_rps": len(np.unique(groups_train[cal_idx])),
        })

        # Covariance-aware ellipse.
        q_ellipse = conformal_quantile(cal_maha, confidence)
        ellipse_covered = test_maha <= q_ellipse

        rows.append({
            "dataset": dataset,
            "protocol": "strict_grouped_split",
            "predictor": "RF300-F0.7",
            "region": "ellipse",
            "confidence": confidence,
            "empirical_coverage": float(np.mean(ellipse_covered)),
            "area_m2": ellipse_area(q_ellipse, Sigma),
            "q": q_ellipse,
            "qx_m": np.nan,
            "qy_m": np.nan,
            "fit_rows": len(fit_idx),
            "calibration_rows": len(cal_idx),
            "fit_rps": len(np.unique(groups_train[fit_idx])),
            "calibration_rps": len(np.unique(groups_train[cal_idx])),
        })

        # Same predictor and scores, RP-corrected finite-sample level.
        n_cal_rps = len(np.unique(groups_train[cal_idx]))
        for region, cal_scores, test_scores, area_fn in (
            ("circle", cal_radial_prop, test_radial_prop,
             lambda q: float(math.pi * q ** 2)),
            ("ellipse", cal_maha, test_maha,
             lambda q: ellipse_area(q, Sigma)),
        ):
            q_rp = conformal_quantile_rp_corrected(cal_scores, confidence, n_cal_rps)
            rows.append({
                "dataset": dataset,
                "protocol": "strict_grouped_split_rp_corrected",
                "predictor": "RF300-F0.7",
                "region": region,
                "confidence": confidence,
                "empirical_coverage": float(np.mean(test_scores <= q_rp)),
                "area_m2": area_fn(q_rp),
                "q": q_rp,
                "qx_m": np.nan,
                "qy_m": np.nan,
                "fit_rows": len(fit_idx),
                "calibration_rows": len(cal_idx),
                "fit_rps": len(np.unique(groups_train[fit_idx])),
                "calibration_rps": n_cal_rps,
            })

    return pd.DataFrame(rows), artifacts


# =============================================================================
# FULL-TRAIN GROUPED-OOF EMPIRICAL UNCERTAINTY
# =============================================================================

def full_oof_empirical_uncertainty(
    dataset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    grid_size_m: float,
    full_proposed_model,
    full_proposed_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Data-efficient OOF empirical calibration.

    NOTE:
    This is intentionally NOT labelled as ordinary split-conformal.
    """
    oof_pred, fold_ids, fold_timing = grouped_oof_predictions(
        X_train,
        y_train,
        groups_train,
        max_features=PROPOSED_MAX_FEATURES,
        n_splits=OOF_FOLDS,
    )

    oof_resid_m = physical_residuals(
        y_train,
        oof_pred,
        grid_size_m,
    )

    test_resid_m = physical_residuals(
        y_test,
        full_proposed_pred,
        grid_size_m,
    )

    oof_radial = np.linalg.norm(oof_resid_m, axis=1)
    test_radial = np.linalg.norm(test_resid_m, axis=1)

    Sigma = regularized_covariance(oof_resid_m)

    oof_maha = mahalanobis_scores(oof_resid_m, Sigma)
    test_maha = mahalanobis_scores(test_resid_m, Sigma)

    rows = []

    for confidence in CONFIDENCE_LEVELS:
        q_circle = conformal_quantile(oof_radial, confidence)
        q_ellipse = conformal_quantile(oof_maha, confidence)

        rows.append({
            "dataset": dataset,
            "protocol": "full_train_grouped_oof_empirical",
            "predictor": "RF300-F0.7 trained on 100% TRAIN",
            "region": "circle",
            "confidence": confidence,
            "empirical_coverage": float(np.mean(test_radial <= q_circle)),
            "area_m2": float(math.pi * q_circle ** 2),
            "q": q_circle,
            "final_train_rows": len(y_train),
            "final_train_rps": len(np.unique(groups_train)),
            "note": "OOF empirical; not ordinary split-conformal guarantee",
        })

        rows.append({
            "dataset": dataset,
            "protocol": "full_train_grouped_oof_empirical",
            "predictor": "RF300-F0.7 trained on 100% TRAIN",
            "region": "ellipse",
            "confidence": confidence,
            "empirical_coverage": float(np.mean(test_maha <= q_ellipse)),
            "area_m2": ellipse_area(q_ellipse, Sigma),
            "q": q_ellipse,
            "final_train_rows": len(y_train),
            "final_train_rps": len(np.unique(groups_train)),
            "note": "OOF empirical; not ordinary split-conformal guarantee",
        })

    oof_df = pd.DataFrame({
        "dataset": dataset,
        "row_index": np.arange(len(y_train)),
        "rp_group_id": groups_train,
        "fold": fold_ids,
        "true_x_grid": y_train[:, 0],
        "true_y_grid": y_train[:, 1],
        "pred_x_grid": oof_pred[:, 0],
        "pred_y_grid": oof_pred[:, 1],
        "dx_m": oof_resid_m[:, 0],
        "dy_m": oof_resid_m[:, 1],
        "radial_error_m": oof_radial,
        "mahalanobis_score": oof_maha,
    })

    summary_df = pd.DataFrame([{
        "dataset": dataset,
        "n_rows": len(oof_df),
        "n_reference_points": len(np.unique(groups_train)),
        "mean_radial_error_m": float(np.mean(oof_radial)),
        "median_radial_error_m": float(np.median(oof_radial)),
        "p80_radial_error_m": float(np.quantile(oof_radial, 0.80)),
        "p90_radial_error_m": float(np.quantile(oof_radial, 0.90)),
        "p95_radial_error_m": float(np.quantile(oof_radial, 0.95)),
        "std_dx_m": float(np.std(oof_resid_m[:, 0], ddof=1)),
        "std_dy_m": float(np.std(oof_resid_m[:, 1], ddof=1)),
        "corr_dx_dy": float(np.corrcoef(oof_resid_m.T)[0, 1]),
        "cov_xx": float(Sigma[0, 0]),
        "cov_xy": float(Sigma[0, 1]),
        "cov_yy": float(Sigma[1, 1]),
    }])

    artifacts = {
        "oof_pred": oof_pred,
        "oof_resid_m": oof_resid_m,
        "oof_radial": oof_radial,
        "oof_maha": oof_maha,
        "Sigma": Sigma,
        "test_resid_m": test_resid_m,
        "test_radial": test_radial,
        "test_maha": test_maha,
    }

    fold_timing = fold_timing.copy()
    fold_timing.insert(0, "dataset", dataset)

    return pd.DataFrame(rows), oof_df, summary_df, artifacts


# =============================================================================
# CLUSTER BOOTSTRAP
# =============================================================================

def paper_metric_from_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_size_m: float,
    idx: np.ndarray,
) -> float:
    residual = y_true[idx] - y_pred[idx]
    coord_rmse = np.sqrt(np.mean(residual ** 2, axis=0)) * grid_size_m
    return float(np.mean(coord_rmse))


def cluster_bootstrap_paired_difference(
    dataset: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    pred_baseline: np.ndarray,
    pred_proposed: np.ndarray,
    grid_size_m: float,
) -> dict:
    """
    Paired cluster bootstrap by TEST reference point.
    Negative difference means proposed is better.
    """
    groups = make_rp_groups(test_df)
    unique_groups = np.unique(groups)

    indices_by_group = {
        g: np.where(groups == g)[0]
        for g in unique_groups
    }

    diffs = []

    rng = np.random.default_rng(SEED)

    for _ in range(BOOTSTRAP_REPEATS):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        idx = np.concatenate([
            indices_by_group[g]
            for g in sampled_groups
        ])

        base = paper_metric_from_indices(
            y_true,
            pred_baseline,
            grid_size_m,
            idx,
        )

        prop = paper_metric_from_indices(
            y_true,
            pred_proposed,
            grid_size_m,
            idx,
        )

        diffs.append(prop - base)

    diffs = np.asarray(diffs, dtype=float)

    return {
        "dataset": dataset,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "difference_definition": "proposed_minus_baseline_m",
        "mean_difference_m": float(np.mean(diffs)),
        "ci95_low_m": float(np.quantile(diffs, 0.025)),
        "ci95_high_m": float(np.quantile(diffs, 0.975)),
        "fraction_proposed_better": float(np.mean(diffs < 0.0)),
    }


# =============================================================================
# DATASET DESCRIPTIVE TABLES / PLOTS
# =============================================================================

def dataset_summary_row(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rtt_cols: list[str],
    rss_cols: list[str],
) -> dict:
    train_coords = coordinate_set(train_df)
    test_coords = coordinate_set(test_df)

    return {
        "dataset": dataset,
        "grid_size_m": GRID_SIZE_M[dataset],
        "n_aps": len(rtt_cols),
        "n_features": len(rtt_cols) + len(rss_cols),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_reference_points": len(train_coords),
        "test_reference_points": len(test_coords),
        "train_test_rp_overlap": len(train_coords & test_coords),
        "published_hybrid_metric_m": PUBLISHED_HYBRID_METRIC_M[dataset],
    }


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    boxes = [
        (0.03, "Hybrid\nRTT + RSS"),
        (0.22, "Subspace RF\n$F=0.7$"),
        (0.42, "RP-grouped\ncross-fitting"),
        (0.63, "Residual\ncovariance"),
        (0.82, "Circle /\nEllipse"),
    ]

    for x, label in boxes:
        ax.text(
            x,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"),
            transform=ax.transAxes,
        )

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.06
        x1 = boxes[i + 1][0] - 0.06
        ax.annotate(
            "",
            xy=(x1, 0.52),
            xytext=(x0, 0.52),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

    ax.text(
        0.5,
        0.12,
        "Final point predictor is retrained on 100% of official TRAIN",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )

    save_figure(fig, "00_method_workflow")


def plot_rp_map(dataset: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_rp = train_df[["X", "Y"]].drop_duplicates().astype(float)
    test_rp = test_df[["X", "Y"]].drop_duplicates().astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(train_rp["X"], train_rp["Y"], s=18, label="Train RPs", alpha=0.7)
    ax.scatter(test_rp["X"], test_rp["Y"], s=26, marker="x", label="Test RPs", alpha=0.8)
    ax.set_xlabel("X grid coordinate")
    ax.set_ylabel("Y grid coordinate")
    ax.set_title(f"{dataset.capitalize()}: train/test reference points")
    ax.legend()

    save_figure(fig, f"{dataset}_dataset_rp_map")


def plot_samples_per_rp(dataset: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_counts = train_df.groupby(["X", "Y"]).size().to_numpy()
    test_counts = test_df.groupby(["X", "Y"]).size().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(30, max(5, len(np.unique(np.concatenate([train_counts, test_counts])))))
    ax.hist(train_counts, bins=bins, alpha=0.6, label="Train RPs")
    ax.hist(test_counts, bins=bins, alpha=0.6, label="Test RPs")
    ax.set_xlabel("Samples per reference point")
    ax.set_ylabel("Number of reference points")
    ax.set_title(f"{dataset.capitalize()}: samples per RP")
    ax.legend()

    save_figure(fig, f"{dataset}_dataset_samples_per_rp")


def plot_ap_missingness(
    dataset: str,
    train_df: pd.DataFrame,
    rtt_cols: list[str],
    rss_cols: list[str],
) -> None:
    RTT = train_df[rtt_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    RSS = train_df[rss_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    rtt_missing = np.mean(np.isnan(RTT) | (RTT == RTT_MISSING), axis=0) * 100.0
    rss_missing = np.mean(np.isnan(RSS) | (RSS == RSS_MISSING), axis=0) * 100.0

    x = np.arange(len(rtt_cols))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(rtt_cols) * 0.7), 5))
    ax.bar(x - width / 2, rtt_missing, width=width, label="RTT unheard")
    ax.bar(x + width / 2, rss_missing, width=width, label="RSS unheard")
    ax.set_xticks(x)
    ax.set_xticklabels([f"AP{i+1}" for i in range(len(rtt_cols))], rotation=45)
    ax.set_ylabel("Missing / unheard samples (%)")
    ax.set_title(f"{dataset.capitalize()}: AP visibility")
    ax.legend()

    save_figure(fig, f"{dataset}_dataset_ap_missingness")


def plot_signal_distributions(
    dataset: str,
    train_df: pd.DataFrame,
    rtt_cols: list[str],
    rss_cols: list[str],
) -> None:
    sample = sample_rows_for_plot(train_df)

    RTT = (
        sample[rtt_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .ravel()
    )
    RSS = (
        sample[rss_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .ravel()
    )

    RTT = RTT[np.isfinite(RTT) & (RTT != RTT_MISSING)]
    RSS = RSS[np.isfinite(RSS) & (RSS != RSS_MISSING)]

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(RTT):
        ax.hist(RTT / 1000.0, bins=60)
    ax.set_xlabel("RTT measurement (m-equivalent from mm field)")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset.capitalize()}: observed RTT distribution")
    save_figure(fig, f"{dataset}_dataset_rtt_distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(RSS):
        ax.hist(RSS, bins=50)
    ax.set_xlabel("RSS (dBm)")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset.capitalize()}: observed RSS distribution")
    save_figure(fig, f"{dataset}_dataset_rss_distribution")


def plot_feature_correlation(
    dataset: str,
    train_df: pd.DataFrame,
    rtt_cols: list[str],
    rss_cols: list[str],
) -> None:
    sample = sample_rows_for_plot(train_df)

    feature_df = sample[rtt_cols + rss_cols].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace(RTT_MISSING, np.nan).replace(RSS_MISSING, np.nan)

    corr = feature_df.corr().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    labels = [f"RTT{i+1}" for i in range(len(rtt_cols))] + [
        f"RSS{i+1}" for i in range(len(rss_cols))
    ]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"{dataset.capitalize()}: feature correlation")
    fig.colorbar(im, ax=ax, label="Pearson correlation")

    save_figure(fig, f"{dataset}_dataset_feature_correlation")


# =============================================================================
# RESULT PLOTS
# =============================================================================

def plot_error_cdf(
    dataset: str,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    grid_size_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, pred in predictions.items():
        errors = np.linalg.norm(
            physical_residuals(y_test, pred, grid_size_m),
            axis=1,
        )
        errors = np.sort(errors)
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=label)

    ax.set_xlabel("Euclidean positioning error (m)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{dataset.capitalize()}: localization-error CDF")
    ax.legend()

    save_figure(fig, f"{dataset}_result_error_cdf")


def plot_error_boxplot(
    dataset: str,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    grid_size_m: float,
) -> None:
    labels = []
    values = []

    for label, pred in predictions.items():
        labels.append(label)
        values.append(
            np.linalg.norm(
                physical_residuals(y_test, pred, grid_size_m),
                axis=1,
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Euclidean positioning error (m)")
    ax.set_title(f"{dataset.capitalize()}: error distribution")

    save_figure(fig, f"{dataset}_result_error_boxplot")


def plot_percentile_profile(
    dataset: str,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    grid_size_m: float,
) -> None:
    percentiles = np.arange(50, 100, 5)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, pred in predictions.items():
        errors = np.linalg.norm(
            physical_residuals(y_test, pred, grid_size_m),
            axis=1,
        )

        vals = [
            np.percentile(errors, p)
            for p in percentiles
        ]

        ax.plot(percentiles, vals, marker="o", label=label)

    ax.set_xlabel("Error percentile")
    ax.set_ylabel("Positioning error (m)")
    ax.set_title(f"{dataset.capitalize()}: tail-error profile")
    ax.legend()

    save_figure(fig, f"{dataset}_result_percentile_profile")


def plot_spatial_error_map(
    dataset: str,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    pred: np.ndarray,
    grid_size_m: float,
    label: str,
) -> None:
    radial = np.linalg.norm(
        physical_residuals(y_test, pred, grid_size_m),
        axis=1,
    )

    plot_df = pd.DataFrame({
        "X": test_df["X"].astype(float).to_numpy(),
        "Y": test_df["Y"].astype(float).to_numpy(),
        "error": radial,
    })

    # Mean error per physical test RP.
    rp = plot_df.groupby(["X", "Y"], as_index=False)["error"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        rp["X"],
        rp["Y"],
        c=rp["error"],
        s=45,
    )
    ax.set_xlabel("X grid coordinate")
    ax.set_ylabel("Y grid coordinate")
    ax.set_title(f"{dataset.capitalize()}: spatial error ({label})")
    fig.colorbar(sc, ax=ax, label="Mean error at RP (m)")

    save_figure(fig, f"{dataset}_result_spatial_error_{label.replace('.', '_')}")


def plot_feature_importance(
    dataset: str,
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    importance = np.asarray(model.feature_importances_, dtype=float)

    df = pd.DataFrame({
        "dataset": dataset,
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    top = df.head(min(20, len(df))).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(top))))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("RF feature importance")
    ax.set_title(f"{dataset.capitalize()}: proposed subspace RF feature importance")

    save_figure(fig, f"{dataset}_result_feature_importance")

    return df


def plot_residual_scatter_with_covariance(
    dataset: str,
    residuals_m: np.ndarray,
    Sigma: np.ndarray,
) -> None:
    sample_idx = np.arange(len(residuals_m))
    if len(sample_idx) > MAX_PLOT_ROWS:
        sample_idx = RNG.choice(
            sample_idx,
            size=MAX_PLOT_ROWS,
            replace=False,
        )

    sample = residuals_m[sample_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sample[:, 0], sample[:, 1], s=7, alpha=0.25)

    # Draw 1-sigma covariance ellipse only to visualize anisotropy.
    width, height, angle = ellipse_axes_angle(1.0, Sigma)
    ell = Ellipse(
        (0.0, 0.0),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2,
        label="1-sigma covariance ellipse",
    )
    ax.add_patch(ell)

    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel(r"$\Delta x$ (m)")
    ax.set_ylabel(r"$\Delta y$ (m)")
    ax.set_title(f"{dataset.capitalize()}: grouped OOF residual geometry")
    ax.legend()

    save_figure(fig, f"{dataset}_uncertainty_oof_residual_scatter")


def plot_oof_residual_cdf(dataset: str, radial: np.ndarray) -> None:
    sorted_e = np.sort(radial)
    cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_e, cdf)
    ax.set_xlabel("Grouped OOF radial residual (m)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{dataset.capitalize()}: OOF residual CDF")

    save_figure(fig, f"{dataset}_uncertainty_oof_residual_cdf")


def plot_example_regions(
    dataset: str,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    Sigma: np.ndarray,
    q_circle: float,
    q_ellipse: float,
    grid_size_m: float,
) -> None:
    """
    Draw a small number of test examples in local physical coordinates.
    Each prediction is shown at (0,0); true position is its physical residual.
    This makes circle/ellipse geometry directly comparable.
    """
    radial = np.linalg.norm(
        physical_residuals(y_test, pred_test, grid_size_m),
        axis=1,
    )

    # Pick examples across the error distribution, not arbitrary random points.
    order = np.argsort(radial)
    positions = np.linspace(
        0,
        len(order) - 1,
        min(N_REGION_EXAMPLES, len(order)),
    ).astype(int)
    chosen = order[positions]

    width, height, angle = ellipse_axes_angle(q_ellipse, Sigma)

    fig, ax = plt.subplots(figsize=(8, 8))

    offset_x = 0.0
    spacing = max(
        2.5 * q_circle,
        1.3 * width,
        1.0,
    )

    for j, idx in enumerate(chosen):
        center = np.array([j * spacing, 0.0])
        resid = physical_residuals(
            y_test[idx:idx+1],
            pred_test[idx:idx+1],
            grid_size_m,
        )[0]

        true_point = center + resid

        circle = Circle(
            center,
            radius=q_circle,
            fill=False,
            linewidth=1.2,
        )
        ax.add_patch(circle)

        ellipse = Ellipse(
            center,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            linestyle="--",
            linewidth=1.2,
        )
        ax.add_patch(ellipse)

        ax.scatter(center[0], center[1], marker="x", s=40)
        ax.scatter(true_point[0], true_point[1], marker="o", s=24)
        ax.plot(
            [center[0], true_point[0]],
            [center[1], true_point[1]],
            linewidth=0.6,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Local physical X offset (m)")
    ax.set_ylabel("Local physical Y offset (m)")
    ax.set_title(
        f"{dataset.capitalize()}: 90% circle vs ellipse examples\n"
        "x = prediction, dot = true position"
    )

    save_figure(fig, f"{dataset}_uncertainty_example_regions_90")



def plot_data_utilization_ablation(data_utilization_df: pd.DataFrame) -> None:
    """
    Compare the proposed subspace RF when trained on the grouped split FIT RPs
    versus when retrained on 100% of official TRAIN RPs.
    """
    datasets = list(DATASETS.keys())
    x = np.arange(len(datasets))
    width = 0.35

    fit75 = []
    full100 = []

    for d in datasets:
        g = data_utilization_df[data_utilization_df["dataset"] == d]
        fit75.append(
            float(g[g["training_scope"] == "grouped_fit_subset"]["paper_metric_m"].iloc[0])
        )
        full100.append(
            float(g[g["training_scope"] == "100pct_official_train"]["paper_metric_m"].iloc[0])
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, fit75, width=width, label="Grouped fit subset (~75% RPs)")
    ax.bar(x + width / 2, full100, width=width, label="100% official TRAIN RPs")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel("Paper-compatible localization metric (m)")
    ax.set_title("Effect of recovering calibration reference points")
    ax.legend()

    save_figure(fig, "aggregate_data_utilization_75_vs_100")


# =============================================================================
# AGGREGATE PLOTS
# =============================================================================

def plot_paper_metric_comparison(point_results: pd.DataFrame) -> None:
    datasets = list(DATASETS.keys())
    x = np.arange(len(datasets))
    width = 0.25

    baseline_vals = []
    proposed_vals = []
    published_vals = []

    for d in datasets:
        group = point_results[point_results["dataset"] == d]

        baseline_vals.append(
            float(group[group["model"] == "RF300-F1.0"]["paper_metric_m"].iloc[0])
        )
        proposed_vals.append(
            float(group[group["model"] == "RF300-F0.7"]["paper_metric_m"].iloc[0])
        )
        published_vals.append(PUBLISHED_HYBRID_METRIC_M[d])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, published_vals, width=width, label="Published hybrid RF")
    ax.bar(x, baseline_vals, width=width, label="Reproduced RF F=1.0")
    ax.bar(x + width, proposed_vals, width=width, label="Proposed RF F=0.7")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel("Paper-compatible localization metric (m)")
    ax.set_title("Published baseline, reproduction, and proposed subspace RF")
    ax.legend()

    save_figure(fig, "aggregate_paper_metric_comparison")


def plot_cv_fold_comparison(cv_fold_results: pd.DataFrame) -> None:
    for dataset in DATASETS:
        group = cv_fold_results[cv_fold_results["dataset"] == dataset]

        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name in ["RF300-F1.0", "RF300-F0.7"]:
            g = group[group["model"] == model_name].sort_values("fold")
            ax.plot(
                g["fold"],
                g["paper_metric_m"],
                marker="o",
                label=model_name,
            )

        ax.set_xlabel("Grouped CV fold")
        ax.set_ylabel("Paper-compatible metric (m)")
        ax.set_title(f"{dataset.capitalize()}: RP-grouped CV")
        ax.legend()

        save_figure(fig, f"{dataset}_result_grouped_cv_folds")


def plot_compute_comparison(point_results: pd.DataFrame) -> None:
    for metric, ylabel, stem in [
        ("train_time_s", "Training time (s)", "aggregate_compute_training"),
        ("predict_time_s", "Inference time (s)", "aggregate_compute_inference"),
        ("model_size_mb", "Serialized model size (MB)", "aggregate_compute_model_size"),
    ]:
        datasets = list(DATASETS.keys())
        x = np.arange(len(datasets))
        width = 0.35

        base = []
        prop = []

        for d in datasets:
            group = point_results[point_results["dataset"] == d]
            base.append(float(group[group["model"] == "RF300-F1.0"][metric].iloc[0]))
            prop.append(float(group[group["model"] == "RF300-F0.7"][metric].iloc[0]))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, base, width=width, label="RF300-F1.0")
        ax.bar(x + width / 2, prop, width=width, label="RF300-F0.7")
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in datasets])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()

        save_figure(fig, stem)


def plot_uncertainty_coverage_area(
    split_df: pd.DataFrame,
    oof_df: pd.DataFrame,
) -> None:
    combined = pd.concat(
        [
            split_df.assign(
                source=np.where(
                    split_df["protocol"] == "strict_grouped_split_rp_corrected",
                    "Strict split (RP-corrected)",
                    "Strict split",
                )
            ),
            oof_df.assign(source="Full-train OOF empirical"),
        ],
        ignore_index=True,
        sort=False,
    )

    # Coverage vs nominal confidence.
    for dataset in DATASETS:
        group = combined[combined["dataset"] == dataset]

        fig, ax = plt.subplots(figsize=(9, 6))

        for (source, predictor, region), g in group.groupby(
            ["source", "predictor", "region"],
            observed=True,
        ):
            label = f"{source}: {predictor} {region}"
            ax.plot(
                g["confidence"] * 100.0,
                g["empirical_coverage"] * 100.0,
                marker="o",
                label=label,
            )

        min_conf = min(CONFIDENCE_LEVELS) * 100.0
        max_conf = max(CONFIDENCE_LEVELS) * 100.0

        ax.plot(
            [min_conf, max_conf],
            [min_conf, max_conf],
            linestyle="--",
            label="Ideal coverage",
        )

        ax.set_xlabel("Nominal confidence (%)")
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_title(f"{dataset.capitalize()}: nominal vs empirical coverage")
        ax.legend(fontsize=7)

        save_figure(fig, f"{dataset}_uncertainty_nominal_vs_coverage")

        # Coverage-area tradeoff.
        fig, ax = plt.subplots(figsize=(9, 6))

        for (source, predictor, region), g in group.groupby(
            ["source", "predictor", "region"],
            observed=True,
        ):
            ax.scatter(
                g["area_m2"],
                g["empirical_coverage"] * 100.0,
                s=55,
                label=f"{source}: {predictor} {region}",
            )

            for _, row in g.iterrows():
                ax.annotate(
                    f"{int(row['confidence']*100)}%",
                    (row["area_m2"], row["empirical_coverage"] * 100.0),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

        ax.set_xlabel("Prediction-region area (m$^2$)")
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_title(f"{dataset.capitalize()}: coverage-area tradeoff")
        ax.legend(fontsize=7)

        save_figure(fig, f"{dataset}_uncertainty_coverage_vs_area")


def plot_circle_ellipse_area(split_df: pd.DataFrame) -> None:
    """
    Directly compare subspace-RF strict split circle vs ellipse.
    """
    for dataset in DATASETS:
        g = split_df[
            (split_df["dataset"] == dataset)
            &
            (split_df["protocol"] == "strict_grouped_split")
            &
            (split_df["predictor"] == "RF300-F0.7")
            &
            (split_df["region"].isin(["circle", "ellipse"]))
        ].copy()

        if g.empty:
            continue

        confidences = sorted(g["confidence"].unique())
        x = np.arange(len(confidences))
        width = 0.35

        circle_vals = [
            float(
                g[
                    (g["confidence"] == c)
                    &
                    (g["region"] == "circle")
                ]["area_m2"].iloc[0]
            )
            for c in confidences
        ]

        ellipse_vals = [
            float(
                g[
                    (g["confidence"] == c)
                    &
                    (g["region"] == "ellipse")
                ]["area_m2"].iloc[0]
            )
            for c in confidences
        ]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - width / 2, circle_vals, width=width, label="Circle")
        ax.bar(x + width / 2, ellipse_vals, width=width, label="Ellipse")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(c*100)}%" for c in confidences])
        ax.set_ylabel("Region area (m$^2$)")
        ax.set_title(f"{dataset.capitalize()}: strict split circle vs ellipse")
        ax.legend()

        save_figure(fig, f"{dataset}_uncertainty_circle_vs_ellipse_area")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 118)
    print("PAPER EXPERIMENT: HYBRID RTT+RSS + SUBSPACE RF + RP GROUPING + CIRCLE/ELLIPSE")
    print("=" * 118)
    print(f"Dataset directory : {DATASET_DIR.resolve()}")
    print(f"Output directory  : {OUT_DIR.resolve()}")
    print(f"RF trees          : {N_TREES}")
    print(f"Baseline F        : {BASELINE_MAX_FEATURES}")
    print(f"Proposed F        : {PROPOSED_MAX_FEATURES}")
    print(f"Grouped CV folds  : {CV_FOLDS}")
    print(f"OOF folds         : {OOF_FOLDS}")
    print(f"Calibration frac. : {CALIBRATION_FRACTION:.2f}")

    plot_workflow()

    dataset_rows = []
    point_rows = []
    published_rows = []
    cv_fold_frames = []
    split_frames = []
    full_oof_uncertainty_frames = []
    oof_residual_frames = []
    oof_summary_frames = []
    oof_timing_frames = []
    feature_importance_frames = []
    bootstrap_rows = []
    data_utilization_rows = []

    for dataset in DATASETS:
        print("\n" + "-" * 118)
        print(dataset.upper())
        print("-" * 118)

        grid_size_m = GRID_SIZE_M[dataset]

        train_df, test_df = load_dataset(dataset)
        rtt_cols, rss_cols = detect_feature_columns(train_df)

        X_train, y_train, feature_names = extract_features_target(
            train_df,
            rtt_cols,
            rss_cols,
        )
        X_test, y_test, _ = extract_features_target(
            test_df,
            rtt_cols,
            rss_cols,
        )

        groups_train = make_rp_groups(train_df)

        ds_row = dataset_summary_row(
            dataset,
            train_df,
            test_df,
            rtt_cols,
            rss_cols,
        )
        dataset_rows.append(ds_row)

        print(
            f"Rows train/test: {len(train_df)}/{len(test_df)} | "
            f"RPs train/test: {ds_row['train_reference_points']}/"
            f"{ds_row['test_reference_points']} | "
            f"RP overlap: {ds_row['train_test_rp_overlap']} | "
            f"APs: {ds_row['n_aps']}"
        )

        # ---------------------------------------------------------------------
        # Dataset figures
        # ---------------------------------------------------------------------
        plot_rp_map(dataset, train_df, test_df)
        plot_samples_per_rp(dataset, train_df, test_df)
        plot_ap_missingness(dataset, train_df, rtt_cols, rss_cols)
        plot_signal_distributions(dataset, train_df, rtt_cols, rss_cols)
        plot_feature_correlation(dataset, train_df, rtt_cols, rss_cols)

        # ---------------------------------------------------------------------
        # Full-train point localization: apples-to-apples RF ablation
        # ---------------------------------------------------------------------
        print("Training full-data baseline RF300-F1.0...")
        base_row, pred_base, model_base = evaluate_full_train_model(
            dataset,
            "RF300-F1.0",
            BASELINE_MAX_FEATURES,
            X_train,
            y_train,
            X_test,
            y_test,
            grid_size_m,
        )

        print("Training full-data proposed RF300-F0.7...")
        prop_row, pred_prop, model_prop = evaluate_full_train_model(
            dataset,
            "RF300-F0.7",
            PROPOSED_MAX_FEATURES,
            X_train,
            y_train,
            X_test,
            y_test,
            grid_size_m,
        )

        base_row["paper_metric_change_vs_F1_pct"] = 0.0
        prop_row["paper_metric_change_vs_F1_pct"] = pct_change(
            prop_row["paper_metric_m"],
            base_row["paper_metric_m"],
        )

        base_row["p90_change_vs_F1_pct"] = 0.0
        prop_row["p90_change_vs_F1_pct"] = pct_change(
            prop_row["p90_euclidean_m"],
            base_row["p90_euclidean_m"],
        )

        point_rows.extend([base_row, prop_row])

        published_value = PUBLISHED_HYBRID_METRIC_M[dataset]

        published_rows.append({
            "dataset": dataset,
            "published_hybrid_metric_m": published_value,
            "reproduced_RF300_F1_metric_m": base_row["paper_metric_m"],
            "proposed_RF300_F07_metric_m": prop_row["paper_metric_m"],
            "reproduction_minus_published_m":
                base_row["paper_metric_m"] - published_value,
            "proposed_minus_published_m":
                prop_row["paper_metric_m"] - published_value,
            "proposed_change_vs_reproduction_pct":
                pct_change(prop_row["paper_metric_m"], base_row["paper_metric_m"]),
            "proposed_change_vs_published_pct":
                pct_change(prop_row["paper_metric_m"], published_value),
        })

        print(
            f"Paper-compatible metric (m): "
            f"published={published_value:.4f}, "
            f"F1.0={base_row['paper_metric_m']:.4f}, "
            f"F0.7={prop_row['paper_metric_m']:.4f}"
        )

        # ---------------------------------------------------------------------
        # Cluster-bootstrap confidence interval for paired improvement.
        # ---------------------------------------------------------------------
        bootstrap_rows.append(
            cluster_bootstrap_paired_difference(
                dataset,
                test_df,
                y_test,
                pred_base,
                pred_prop,
                grid_size_m,
            )
        )

        # ---------------------------------------------------------------------
        # RP-grouped CV confirmation.
        # ---------------------------------------------------------------------
        print("Running RP-grouped CV...")
        cv_fold_df = grouped_cv_compare(
            dataset,
            X_train,
            y_train,
            groups_train,
            grid_size_m,
        )
        cv_fold_frames.append(cv_fold_df)

        # ---------------------------------------------------------------------
        # Strict grouped split-conformal comparison.
        # ---------------------------------------------------------------------
        print("Running strict grouped split-conformal circle/ellipse experiment...")
        split_df, split_artifacts = strict_split_conformal_experiment(
            dataset,
            X_train,
            y_train,
            groups_train,
            X_test,
            y_test,
            grid_size_m,
        )
        split_frames.append(split_df)

        # ---------------------------------------------------------------------
        # Data-utilization ablation:
        # same proposed F=0.7 predictor, but ~75% grouped FIT RPs vs 100% TRAIN.
        # This isolates the benefit of recovering calibration RPs.
        # ---------------------------------------------------------------------
        split_prop_metrics = localization_metrics(
            y_test,
            split_artifacts["pred_test_proposed"],
            grid_size_m,
        )

        data_utilization_rows.append({
            "dataset": dataset,
            "training_scope": "grouped_fit_subset",
            "train_rows": len(split_artifacts["fit_idx"]),
            "train_rps": len(np.unique(groups_train[split_artifacts["fit_idx"]])),
            **split_prop_metrics,
        })

        data_utilization_rows.append({
            "dataset": dataset,
            "training_scope": "100pct_official_train",
            "train_rows": len(y_train),
            "train_rps": len(np.unique(groups_train)),
            **localization_metrics(y_test, pred_prop, grid_size_m),
        })

        # ---------------------------------------------------------------------
        # Full-training OOF empirical calibration.
        # ---------------------------------------------------------------------
        print("Generating full-training RP-grouped OOF residuals...")
        (
            full_oof_uncertainty_df,
            oof_residual_df,
            oof_summary_df,
            oof_artifacts,
        ) = full_oof_empirical_uncertainty(
            dataset,
            X_train,
            y_train,
            groups_train,
            X_test,
            y_test,
            grid_size_m,
            model_prop,
            pred_prop,
        )

        full_oof_uncertainty_frames.append(full_oof_uncertainty_df)
        oof_residual_frames.append(oof_residual_df)
        oof_summary_frames.append(oof_summary_df)

        # OOF timings are already embedded in helper output via grouped OOF;
        # full detailed timing table is not separately returned here to avoid
        # repeating another expensive OOF run.

        # ---------------------------------------------------------------------
        # Feature importance.
        # ---------------------------------------------------------------------
        fi_df = plot_feature_importance(
            dataset,
            model_prop,
            feature_names,
        )
        feature_importance_frames.append(fi_df)

        # ---------------------------------------------------------------------
        # Result figures.
        # ---------------------------------------------------------------------
        plot_error_cdf(
            dataset,
            y_test,
            {
                "RF300-F1.0": pred_base,
                "RF300-F0.7": pred_prop,
            },
            grid_size_m,
        )

        plot_error_boxplot(
            dataset,
            y_test,
            {
                "RF300-F1.0": pred_base,
                "RF300-F0.7": pred_prop,
            },
            grid_size_m,
        )

        plot_percentile_profile(
            dataset,
            y_test,
            {
                "RF300-F1.0": pred_base,
                "RF300-F0.7": pred_prop,
            },
            grid_size_m,
        )

        plot_spatial_error_map(
            dataset,
            test_df,
            y_test,
            pred_base,
            grid_size_m,
            "RF300-F1.0",
        )

        plot_spatial_error_map(
            dataset,
            test_df,
            y_test,
            pred_prop,
            grid_size_m,
            "RF300-F0.7",
        )

        plot_residual_scatter_with_covariance(
            dataset,
            oof_artifacts["oof_resid_m"],
            oof_artifacts["Sigma"],
        )

        plot_oof_residual_cdf(
            dataset,
            oof_artifacts["oof_radial"],
        )

        # Example 90% OOF empirical regions.
        oof90 = full_oof_uncertainty_df[
            full_oof_uncertainty_df["confidence"] == 0.90
        ]

        q_circle_90 = float(
            oof90[oof90["region"] == "circle"]["q"].iloc[0]
        )
        q_ellipse_90 = float(
            oof90[oof90["region"] == "ellipse"]["q"].iloc[0]
        )

        plot_example_regions(
            dataset,
            test_df,
            y_test,
            pred_prop,
            oof_artifacts["Sigma"],
            q_circle_90,
            q_ellipse_90,
            grid_size_m,
        )

        # Free large baseline model; proposed model is no longer needed after plots.
        del model_base
        del model_prop
        gc.collect()

    # =========================================================================
    # AGGREGATE TABLES
    # =========================================================================
    dataset_summary_df = pd.DataFrame(dataset_rows)
    point_results_df = pd.DataFrame(point_rows)
    published_comparison_df = pd.DataFrame(published_rows)
    cv_fold_results_df = pd.concat(cv_fold_frames, ignore_index=True)

    is_pooled = cv_fold_results_df["fold"].astype(str) == "pooled"
    per_fold = cv_fold_results_df[~is_pooled]

    cv_summary_df = (
        cv_fold_results_df[is_pooled][[
            "dataset", "model", "paper_metric_m", "rmse_euclidean_m",
            "mean_euclidean_m", "p90_euclidean_m", "p95_euclidean_m",
        ]]
        .merge(
            per_fold
            .groupby(["dataset", "model"], as_index=False, observed=True)
            .agg(
                paper_metric_fold_std_m=("paper_metric_m", "std"),
                train_time_s=("train_time_s", "mean"),
                predict_time_s=("predict_time_s", "mean"),
            ),
            on=["dataset", "model"],
        )
    )

    split_results_df = pd.concat(split_frames, ignore_index=True)
    full_oof_uncertainty_df = pd.concat(
        full_oof_uncertainty_frames,
        ignore_index=True,
    )
    oof_residuals_df = pd.concat(oof_residual_frames, ignore_index=True)
    oof_summary_df = pd.concat(oof_summary_frames, ignore_index=True)
    feature_importance_df = pd.concat(
        feature_importance_frames,
        ignore_index=True,
    )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    data_utilization_df = pd.DataFrame(data_utilization_rows)

    # Add within-dataset improvement from recovering all official TRAIN RPs.
    data_utilization_df["change_vs_grouped_fit_pct"] = np.nan
    for dataset in DATASETS:
        mask = data_utilization_df["dataset"] == dataset
        g = data_utilization_df[mask]
        fit_value = float(
            g[g["training_scope"] == "grouped_fit_subset"]["paper_metric_m"].iloc[0]
        )
        full_idx = g[g["training_scope"] == "100pct_official_train"].index[0]
        data_utilization_df.loc[full_idx, "change_vs_grouped_fit_pct"] = pct_change(
            data_utilization_df.loc[full_idx, "paper_metric_m"],
            fit_value,
        )
        fit_idx_row = g[g["training_scope"] == "grouped_fit_subset"].index[0]
        data_utilization_df.loc[fit_idx_row, "change_vs_grouped_fit_pct"] = 0.0

    # Contribution summary for quick manuscript writing.
    contribution_rows = []

    for dataset in DATASETS:
        point_group = point_results_df[point_results_df["dataset"] == dataset]
        base = point_group[point_group["model"] == "RF300-F1.0"].iloc[0]
        prop = point_group[point_group["model"] == "RF300-F0.7"].iloc[0]

        split_group = split_results_df[
            (split_results_df["dataset"] == dataset)
            &
            (split_results_df["protocol"] == "strict_grouped_split")
            &
            (split_results_df["predictor"] == "RF300-F0.7")
            &
            (split_results_df["confidence"] == 0.90)
        ]

        circle = split_group[split_group["region"] == "circle"].iloc[0]
        ellipse = split_group[split_group["region"] == "ellipse"].iloc[0]

        oof_group = full_oof_uncertainty_df[
            (full_oof_uncertainty_df["dataset"] == dataset)
            &
            (full_oof_uncertainty_df["confidence"] == 0.90)
        ]

        oof_circle = oof_group[oof_group["region"] == "circle"].iloc[0]
        oof_ellipse = oof_group[oof_group["region"] == "ellipse"].iloc[0]

        util_group = data_utilization_df[data_utilization_df["dataset"] == dataset]
        util_fit = util_group[
            util_group["training_scope"] == "grouped_fit_subset"
        ].iloc[0]
        util_full = util_group[
            util_group["training_scope"] == "100pct_official_train"
        ].iloc[0]

        contribution_rows.append({
            "dataset": dataset,
            "published_metric_m": PUBLISHED_HYBRID_METRIC_M[dataset],
            "reproduced_F1_metric_m": base["paper_metric_m"],
            "proposed_F07_metric_m": prop["paper_metric_m"],
            "F07_improvement_vs_F1_pct":
                -pct_change(prop["paper_metric_m"], base["paper_metric_m"]),
            "F07_improvement_vs_published_pct":
                -pct_change(prop["paper_metric_m"], PUBLISHED_HYBRID_METRIC_M[dataset]),
            "F07_grouped_fit_metric_m": util_fit["paper_metric_m"],
            "F07_full_train_metric_m": util_full["paper_metric_m"],
            "full_train_improvement_vs_grouped_fit_pct":
                -pct_change(util_full["paper_metric_m"], util_fit["paper_metric_m"]),
            "strict90_circle_coverage": circle["empirical_coverage"],
            "strict90_circle_area_m2": circle["area_m2"],
            "strict90_ellipse_coverage": ellipse["empirical_coverage"],
            "strict90_ellipse_area_m2": ellipse["area_m2"],
            "strict90_ellipse_area_change_vs_circle_pct":
                pct_change(ellipse["area_m2"], circle["area_m2"]),
            "oof90_circle_coverage": oof_circle["empirical_coverage"],
            "oof90_circle_area_m2": oof_circle["area_m2"],
            "oof90_ellipse_coverage": oof_ellipse["empirical_coverage"],
            "oof90_ellipse_area_m2": oof_ellipse["area_m2"],
            "oof90_ellipse_area_change_vs_circle_pct":
                pct_change(oof_ellipse["area_m2"], oof_circle["area_m2"]),
        })

    contribution_summary_df = pd.DataFrame(contribution_rows)

    # =========================================================================
    # SAVE TABLES
    # =========================================================================
    save_table(dataset_summary_df, "dataset_summary.csv")
    save_table(point_results_df, "point_localization_results.csv")
    save_table(published_comparison_df, "published_comparison.csv")
    save_table(cv_fold_results_df, "grouped_cv_fold_results.csv")
    save_table(cv_summary_df, "grouped_cv_summary.csv")
    save_table(split_results_df, "split_conformal_results.csv")
    save_table(
        full_oof_uncertainty_df,
        "full_oof_uncertainty_results.csv",
    )
    save_table(oof_residuals_df, "grouped_oof_residuals.csv", latex=False)
    save_table(oof_summary_df, "oof_residual_summary.csv")
    save_table(feature_importance_df, "feature_importance.csv", latex=False)
    save_table(bootstrap_df, "cluster_bootstrap_comparison.csv")
    save_table(data_utilization_df, "data_utilization_ablation.csv")
    save_table(contribution_summary_df, "contribution_summary.csv")

    # =========================================================================
    # AGGREGATE PLOTS
    # =========================================================================
    plot_paper_metric_comparison(point_results_df)
    plot_cv_fold_comparison(per_fold)
    plot_compute_comparison(point_results_df)
    plot_data_utilization_ablation(data_utilization_df)
    plot_uncertainty_coverage_area(
        split_results_df,
        full_oof_uncertainty_df,
    )
    plot_circle_ellipse_area(split_results_df)

    # =========================================================================
    # MACHINE-READABLE EXPERIMENT METADATA
    # =========================================================================
    metadata = {
        "seed": SEED,
        "n_trees": N_TREES,
        "baseline_max_features": BASELINE_MAX_FEATURES,
        "proposed_max_features": PROPOSED_MAX_FEATURES,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "cv_folds": CV_FOLDS,
        "oof_folds": OOF_FOLDS,
        "calibration_fraction": CALIBRATION_FRACTION,
        "confidence_levels": CONFIDENCE_LEVELS,
        "grid_size_m": GRID_SIZE_M,
        "published_hybrid_metric_m": PUBLISHED_HYBRID_METRIC_M,
        "rtt_missing": RTT_MISSING,
        "rss_missing": RSS_MISSING,
        "scientific_note": (
            "Strict grouped split-conformal and full-train grouped-OOF empirical "
            "calibration are reported separately. OOF empirical calibration is "
            "not labelled as ordinary split-conformal finite-sample guarantee."
        ),
    }

    (OUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================
    print("\n" + "=" * 118)
    print("PAPER-READY CONTRIBUTION SUMMARY")
    print("=" * 118)

    display_cols = [
        "dataset",
        "published_metric_m",
        "reproduced_F1_metric_m",
        "proposed_F07_metric_m",
        "F07_improvement_vs_F1_pct",
        "F07_improvement_vs_published_pct",
        "F07_grouped_fit_metric_m",
        "F07_full_train_metric_m",
        "full_train_improvement_vs_grouped_fit_pct",
        "strict90_circle_coverage",
        "strict90_circle_area_m2",
        "strict90_ellipse_coverage",
        "strict90_ellipse_area_m2",
        "strict90_ellipse_area_change_vs_circle_pct",
        "oof90_circle_coverage",
        "oof90_ellipse_coverage",
    ]

    print(
        contribution_summary_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nBootstrap paired point-localization comparison:")
    print(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\nAll tables:")
    for p in sorted(TABLE_DIR.glob("*.csv")):
        print(f"  {p}")

    print("\nAll plots:")
    for p in sorted(PLOT_DIR.glob("*.png")):
        print(f"  {p}")

    print("\nSCIENTIFIC REMINDER:")
    print(
        "  'strict_grouped_split' is the clean split-conformal experiment.\n"
        "  'full_train_grouped_oof_empirical' uses all TRAIN RPs but must not be\n"
        "  described as ordinary split-conformal with an exact finite-sample\n"
        "  guarantee unless a formally justified cross-conformal/CV+ construction\n"
        "  is adopted in the final methodology."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
