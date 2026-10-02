"""
eetac_external_validation_subspace_rf.py
========================================

External validation of our hybrid RTT+RSS Subspace Random Forest on the
EETAC Auditorium dataset used by the recent IEEE IoT Journal work.

PURPOSE
-------
This script answers a focused question:

    Does RF(max_features=0.7) still improve localization when evaluated
    on a completely different public RTT/RSSI dataset?

It intentionally separates three evaluation protocols:

1) STRATIFIED ROW CV
   - Samples from every physical RP can appear in both train and validation.
   - This is closer to fingerprint-recognition evaluation used by many papers.
   - Primary metric: true 2D RMSE in metres.

2) STRICT RP-GROUPED CV
   - Entire physical reference points are held out.
   - No scans from one (x,y) location can occur in both train and validation.
   - This tests spatial generalization to unseen RPs.

3) CROSS-DAY GENERALIZATION
   - Train Day 1 -> Test Day 2.
   - Train Day 2 -> Test Day 1.
   - Tests temporal robustness while keeping the same physical RP layout.

MODELS
------
Baseline:
    RandomForestRegressor(n_estimators=300, max_features=1.0)

Proposed:
    RandomForestRegressor(n_estimators=300, max_features=0.7)

No model selection is performed on the test folds.  Both RF configurations
are fixed before evaluation.

DATA
----
Expected under ./dataset:

    EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 1.csv
    EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 1.csv
    EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 2.csv
    EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 2.csv

The coordinates x,y are already in metres, so NO grid-size multiplication
is used here.

PHYSICAL AP HANDLING
--------------------
The raw files expose BSSIDs/radio interfaces.  We aggregate those BSSIDs
into physical APs.

Google physical APs:
    G1 = 58:cb:52:d7:b4:d9
    G2 = 58:cb:52:d7:b5:31
    G3 = 58:cb:52:d7:b5:5d
    G4 = 58:cb:52:d7:bb:61

Linksys/Belkin candidates are grouped by radio-family BSSIDs:
    L1 = 30:23:03:87:17:80/81/82 + 36:23:03:87:17:80
    L2 = 30:23:03:87:17:98/99/9a + 36:23:03:87:17:98
    L3 = 30:23:03:87:19:6c/6d/6e + 36:23:03:87:19:6c
    L4 = c4:41:1e:fa:08:88/89/8a + ca:41:1e:fa:08:88

For one physical AP:
    RSS feature = strongest available RSS among its BSSIDs.
    RTT feature = median available RTT among its BSSIDs.

Missing value in this dataset is -100.

SCENARIOS
---------
Medium:
    all four Google APs -- unambiguous and directly useful.

Low:
    the IEEE paper uses two opposite-corner Google APs.
    The raw CSV does not label Google 1..4 explicitly.
    To avoid cherry-picking, BOTH opposite-corner pairs are evaluated.

High:
    IEEE uses four Google APs plus three Linksys APs:
    both long-wall centres + one short-wall centre.
    The raw CSV has candidates on both short walls.
    BOTH topology-consistent variants are evaluated.

All8:
    exploratory only; all four Google + all four Linksys physical APs.

IMPORTANT
---------
Do NOT select Low/High variant after looking at accuracy and call it the
IEEE configuration.  Report both variants unless the exact AP label mapping
is independently confirmed from the original experiment metadata/figure.

OUTPUT
------
eetac_external_validation_results/
    dataset_summary.csv
    physical_ap_inferred_locations.csv
    scenario_definition.csv
    fold_results.csv
    summary_results.csv
    cross_day_results.csv
    paired_improvement_summary.csv
    plots/*.png
    plots/*.pdf

Dependencies:
    numpy
    pandas
    matplotlib
    scikit-learn

Run:
    python eetac_external_validation_subspace_rf.py
"""

from __future__ import annotations

import gc
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, StratifiedKFold


# =============================================================================
# CONFIG
# =============================================================================

SEED = 42
DATASET_DIR = Path("dataset")
OUT_DIR = Path("eetac_external_validation_results")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

N_TREES = 300
BASELINE_MAX_FEATURES = 1.0
PROPOSED_MAX_FEATURES = 0.7
N_FOLDS = 5
MISSING = -100.0
PREDICTION_REPEATS = 5
FIG_DPI = 300

# The EETAC auditorium uses a 5x5 physical RP grid.  Day 1 contains
# the top row as y=8.1 m while Day 2 uses y=8.0 m for the same physical
# locations.  Snap only small coordinate deviations to the canonical grid
# so the same physical RP is not treated as two different groups.
CANONICAL_X = np.array([0.0, 4.8, 9.6, 14.4, 19.2], dtype=float)
CANONICAL_Y = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=float)
COORD_SNAP_TOL_M = 0.25

# The two acquisition days use mirrored y axes (Day-2 y = 8 - Day-1 y): the
# per-RP Google RTT fingerprints of Day 1 match Day 2 under the mirror for
# 22/25 RPs and under the identity only for the y = 4 m row.  Without this
# alignment every pooled or cross-day EETAC experiment mixes two physical
# locations under one label.  The detection is data-driven and audited in
# day_frame_alignment_audit.csv.
ALIGN_DAY_FRAMES = True
FRAME_CHECK_APS = ["G1", "G2", "G3", "G4"]

FILES = {
    1: {
        "rss": "EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 1.csv",
        "rtt": "EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 1.csv",
    },
    2: {
        "rss": "EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 2.csv",
        "rtt": "EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 2.csv",
    },
}

META_COLS = ["batch", "x", "y", "z", "brand", "model", "angle", "sampleNumber"]

# Physical AP -> base BSSIDs (without #0/#1 suffix).
PHYSICAL_APS = {
    "G1": ["58:cb:52:d7:b4:d9"],
    "G2": ["58:cb:52:d7:b5:31"],
    "G3": ["58:cb:52:d7:b5:5d"],
    "G4": ["58:cb:52:d7:bb:61"],

    "L1": [
        "30:23:03:87:17:80",
        "30:23:03:87:17:81",
        "30:23:03:87:17:82",
        "36:23:03:87:17:80",
    ],
    "L2": [
        "30:23:03:87:17:98",
        "30:23:03:87:17:99",
        "30:23:03:87:17:9a",
        "36:23:03:87:17:98",
    ],
    "L3": [
        "30:23:03:87:19:6c",
        "30:23:03:87:19:6d",
        "30:23:03:87:19:6e",
        "36:23:03:87:19:6c",
    ],
    "L4": [
        "c4:41:1e:fa:08:88",
        "c4:41:1e:fa:08:89",
        "c4:41:1e:fa:08:8a",
        "ca:41:1e:fa:08:88",
    ],
}


# =============================================================================
# HELPERS
# =============================================================================

def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(PLOT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_rf(max_features: float) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_TREES,
        max_features=max_features,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )


def timed_fit(model, X, y) -> float:
    gc.collect()
    t0 = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t0


def timed_predict(model, X) -> tuple[np.ndarray, float]:
    _ = model.predict(X[: min(len(X), 32)])
    times = []
    pred = None
    for _ in range(PREDICTION_REPEATS):
        t0 = time.perf_counter()
        pred = model.predict(X)
        times.append(time.perf_counter() - t0)
    return pred, float(np.median(times))


def metrics_2d(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Coordinates in this dataset are already metres.

    Primary metric is true two-dimensional RMSE:
        sqrt(mean(dx^2 + dy^2))
    """
    residual = y_true - y_pred
    radial = np.linalg.norm(residual, axis=1)

    rmse_x = float(np.sqrt(np.mean(residual[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(residual[:, 1] ** 2)))

    return {
        "rmse_2d_m": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
        "rmse_x_m": rmse_x,
        "rmse_y_m": rmse_y,
        "mean_error_m": float(np.mean(radial)),
        "median_error_m": float(np.median(radial)),
        "p80_error_m": float(np.quantile(radial, 0.80)),
        "p90_error_m": float(np.quantile(radial, 0.90)),
        "p95_error_m": float(np.quantile(radial, 0.95)),
    }


def pct_reduction(proposed: float, baseline: float) -> float:
    return 100.0 * (baseline - proposed) / baseline


def rp_label_from_xy(df: pd.DataFrame) -> np.ndarray:
    labels = (
        df["x"].astype(str).str.strip()
        + "|"
        + df["y"].astype(str).str.strip()
    )
    return labels.to_numpy()


def make_group_ids(df: pd.DataFrame) -> np.ndarray:
    labels, _ = pd.factorize(
        pd.MultiIndex.from_frame(df[["x", "y"]].astype(float))
    )
    return labels.astype(int)


# =============================================================================
# LOAD / MERGE
# =============================================================================

def read_day(day: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rss_path = DATASET_DIR / FILES[day]["rss"]
    rtt_path = DATASET_DIR / FILES[day]["rtt"]

    if not rss_path.exists():
        raise FileNotFoundError(f"Missing: {rss_path}")
    if not rtt_path.exists():
        raise FileNotFoundError(f"Missing: {rtt_path}")

    rss = pd.read_csv(rss_path, low_memory=False)
    rtt = pd.read_csv(rtt_path, low_memory=False)

    # Normalize headers because repository files can differ slightly in
    # whitespace/BOM/casing between days.
    rss.columns = [str(c).replace("\ufeff", "").strip() for c in rss.columns]
    rtt.columns = [str(c).replace("\ufeff", "").strip() for c in rtt.columns]

    for c in META_COLS:
        if c not in rss.columns or c not in rtt.columns:
            raise ValueError(f"Day {day}: required metadata column '{c}' missing.")

    return rss, rtt


def snap_to_canonical(values: pd.Series, canonical: np.ndarray) -> pd.Series:
    """
    Snap only values within COORD_SNAP_TOL_M of a known physical grid value.
    Values farther away are preserved, preventing accidental relabeling.
    """
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = arr.copy()

    for i, value in enumerate(arr):
        if not np.isfinite(value):
            continue
        j = int(np.argmin(np.abs(canonical - value)))
        if abs(canonical[j] - value) <= COORD_SNAP_TOL_M:
            out[i] = canonical[j]

    return pd.Series(out, index=values.index)


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["x", "y", "z", "angle", "sampleNumber"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Preserve the original coordinates for audit/debugging.
    out["x_raw"] = out["x"]
    out["y_raw"] = out["y"]

    # Canonicalize the known 5x5 auditorium RP grid.
    out["x"] = snap_to_canonical(out["x"], CANONICAL_X)
    out["y"] = snap_to_canonical(out["y"], CANONICAL_Y)

    for c in ["batch", "brand", "model"]:
        out[c] = out[c].astype(str)

    return out


def aggregate_physical_ap(
    rss: pd.DataFrame,
    rtt: pd.DataFrame,
    ap_name: str,
    bssids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, list[str], list[str]]:
    """
    Aggregate multiple radio/BSSID interfaces into one physical AP feature.

    Matching is case-insensitive and ignores surrounding whitespace.

    RSS:
        strongest observed radio (maximum dBm).

    RTT:
        median of available radio RTT estimates.

    If a physical AP is absent from either modality on a given day, return
    None for the features and let the caller log/skip the AP instead of
    crashing the whole experiment.
    """
    rss_lookup = {str(c).strip().lower(): c for c in rss.columns}
    rtt_lookup = {str(c).strip().lower(): c for c in rtt.columns}

    rss_cols = []
    rtt_cols = []

    for b in bssids:
        rss_key = f"{b}#0".lower()
        rtt_key = f"{b}#1".lower()

        if rss_key in rss_lookup:
            rss_cols.append(rss_lookup[rss_key])
        if rtt_key in rtt_lookup:
            rtt_cols.append(rtt_lookup[rtt_key])

    # Do not fail: different acquisition days may contain different radios.
    if not rss_cols or not rtt_cols:
        return None, None, rtt_cols, rss_cols

    rss_v = rss[rss_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    rtt_v = rtt[rtt_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    # Sentinel -> NaN for aggregation.
    rss_v = np.where((~np.isfinite(rss_v)) | (rss_v == MISSING), np.nan, rss_v)
    rtt_v = np.where((~np.isfinite(rtt_v)) | (rtt_v == MISSING), np.nan, rtt_v)

    # Some rows may have no observation from a physical AP.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rss_feature = np.nanmax(rss_v, axis=1)
        rtt_feature = np.nanmedian(rtt_v, axis=1)

    rss_feature = np.where(np.isfinite(rss_feature), rss_feature, MISSING)
    rtt_feature = np.where(np.isfinite(rtt_feature), rtt_feature, MISSING)

    return rtt_feature, rss_feature, rtt_cols, rss_cols

def build_day_table(day: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        table: metadata + RTT_<AP> + RSS_<AP>
        audit: AP/BSSID availability table
    """
    rss, rtt = read_day(day)
    rss = normalize_metadata(rss)
    rtt = normalize_metadata(rtt)

    # Verify row alignment robustly.
    left = rss[META_COLS].reset_index(drop=True)
    right = rtt[META_COLS].reset_index(drop=True)

    if len(left) != len(right):
        raise ValueError(
            f"Day {day}: RSS/RTT row counts differ: {len(left)} vs {len(right)}"
        )

    same = np.ones(len(left), dtype=bool)
    for c in META_COLS:
        if c in ["x", "y", "z", "angle", "sampleNumber"]:
            same &= np.isclose(
                pd.to_numeric(left[c], errors="coerce"),
                pd.to_numeric(right[c], errors="coerce"),
                equal_nan=True,
            )
        else:
            same &= left[c].astype(str).to_numpy() == right[c].astype(str).to_numpy()

    if not np.all(same):
        bad = np.where(~same)[0][:10]
        raise ValueError(
            f"Day {day}: RSS/RTT rows do not align at indices {bad.tolist()}."
        )

    table = rss[META_COLS].copy()
    table["day"] = day

    audit_rows = []

    for ap_name, bssids in PHYSICAL_APS.items():
        rtt_feature, rss_feature, rtt_cols, rss_cols = aggregate_physical_ap(
            rss,
            rtt,
            ap_name,
            bssids,
        )
        available = (rtt_feature is not None) and (rss_feature is not None)

        if available:
            table[f"RTT_{ap_name}"] = rtt_feature
            table[f"RSS_{ap_name}"] = rss_feature

        audit_rows.append({
            "day": day,
            "physical_ap": ap_name,
            "available_both_modalities": bool(available),
            "rtt_columns_found": ";".join(map(str, rtt_cols)),
            "rss_columns_found": ";".join(map(str, rss_cols)),
            "rtt_visibility_pct":
                float(np.mean(rtt_feature != MISSING) * 100.0)
                if available else np.nan,
            "rss_visibility_pct":
                float(np.mean(rss_feature != MISSING) * 100.0)
                if available else np.nan,
        })

    return table, pd.DataFrame(audit_rows)


def detect_day_frame(day1: pd.DataFrame, day2: pd.DataFrame) -> pd.DataFrame:
    """
    Compare per-RP median RTT fingerprints across days and count, for each
    candidate coordinate transform of Day 2, how many Day-1 RPs have their
    nearest Day-2 fingerprint at the transformed location.
    """
    cols = [f"RTT_{ap}" for ap in FRAME_CHECK_APS]
    m1 = day1.groupby(["x", "y"])[cols].median()
    m2 = day2.groupby(["x", "y"])[cols].median()
    ymax = float(CANONICAL_Y.max())

    transforms = {
        "identity": lambda x, y: (x, y),
        "mirror_y": lambda x, y: (x, round(ymax - y, 6)),
    }
    hits = {name: 0 for name in transforms}

    for (x, y), row in m1.iterrows():
        dist = ((m2 - row) ** 2).sum(axis=1)
        nx, ny = dist.idxmin()
        for name, fn in transforms.items():
            tx, ty = fn(x, y)
            if np.isclose(nx, tx) and np.isclose(ny, ty):
                hits[name] += 1

    # The y = 4 m row maps to itself under both transforms; it carries no
    # evidence, so exclude it from the decision counts.
    centre = int(np.isclose(m1.index.get_level_values("y"), ymax / 2).sum())
    evidence = {k: v - centre for k, v in hits.items()}
    selected = max(evidence, key=evidence.get)

    return pd.DataFrame([{
        "n_rps": len(m1),
        "identity_matches": hits["identity"],
        "mirror_y_matches": hits["mirror_y"],
        "uninformative_centre_row_rps": centre,
        "selected": selected,
    }])


# =============================================================================
# INFER AP LOCATIONS / BUILD TOPOLOGY SCENARIOS
# =============================================================================

def infer_ap_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer an approximate physical AP location from the RP with the highest
    mean observed RSS for that AP.  This is used ONLY to define topology
    variants; not as an input feature.
    """
    rows = []

    available_aps = [
        ap for ap in PHYSICAL_APS
        if f"RSS_{ap}" in df.columns
    ]

    for ap in available_aps:
        col = f"RSS_{ap}"
        tmp = df[["x", "y", col]].copy()
        tmp[col] = tmp[col].replace(MISSING, np.nan)

        rp_mean = (
            tmp.groupby(["x", "y"], as_index=False)[col]
            .mean()
            .dropna()
            .sort_values(col, ascending=False)
        )

        if rp_mean.empty:
            x_hat = y_hat = mean_rss = np.nan
        else:
            best = rp_mean.iloc[0]
            x_hat = float(best["x"])
            y_hat = float(best["y"])
            mean_rss = float(best[col])

        rows.append({
            "physical_ap": ap,
            "vendor_group": "Google" if ap.startswith("G") else "Linksys",
            "inferred_x_m": x_hat,
            "inferred_y_m": y_hat,
            "best_rp_mean_rss_dbm": mean_rss,
        })

    return pd.DataFrame(rows)


def euclidean(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def build_scenarios(
    ap_locations: pd.DataFrame,
    common_aps: list[str],
) -> dict[str, list[str]]:
    """
    Build scenarios only from physical APs that have BOTH RTT and RSS
    available on BOTH acquisition days.

    This prevents a Day-2 missing BSSID from crashing cross-day validation.
    """
    loc = {
        row.physical_ap: (row.inferred_x_m, row.inferred_y_m)
        for row in ap_locations.itertuples()
        if row.physical_ap in common_aps
    }

    google = [g for g in ["G1", "G2", "G3", "G4"] if g in common_aps]
    linksys = [l for l in ["L1", "L2", "L3", "L4"] if l in common_aps]

    scenarios = {}

    # Medium requires all four Google APs.
    if len(google) == 4:
        scenarios["Medium_4Google"] = google.copy()

        # Two opposite-corner Google pairs.
        pairs = []
        for i in range(len(google)):
            for j in range(i + 1, len(google)):
                a, b = google[i], google[j]
                pairs.append((euclidean(loc[a], loc[b]), a, b))
        pairs.sort(reverse=True)

        diagonal_pairs = []
        used = set()
        for dist, a, b in pairs:
            key = frozenset((a, b))
            if key not in used:
                diagonal_pairs.append([a, b])
                used.add(key)
            if len(diagonal_pairs) == 2:
                break

        if len(diagonal_pairs) >= 1:
            scenarios["Low_diag_A"] = diagonal_pairs[0]
        if len(diagonal_pairs) >= 2:
            scenarios["Low_diag_B"] = diagonal_pairs[1]

    # High-density scenarios.
    if len(google) == 4 and len(linksys) >= 3:
        # If exactly three Linksys are available in both days, that subset
        # is the only reproducible common cross-day High scenario.
        if len(linksys) == 3:
            scenarios["High_4Google_3Linksys"] = google + linksys

        # If all four are common, preserve the two topology-consistent
        # short-wall variants instead of cherry-picking.
        elif len(linksys) >= 4:
            xs = [loc[g][0] for g in google]
            ys = [loc[g][1] for g in google]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            xmid = (xmin + xmax) / 2.0
            ymid = (ymin + ymax) / 2.0

            targets = {
                "bottom_mid": (xmid, ymin),
                "top_mid": (xmid, ymax),
                "left_mid": (xmin, ymid),
                "right_mid": (xmax, ymid),
            }

            remaining = set(linksys)
            wall_assignment = {}
            for wall in ["bottom_mid", "top_mid", "left_mid", "right_mid"]:
                if not remaining:
                    break
                ap = min(
                    remaining,
                    key=lambda a: euclidean(loc[a], targets[wall])
                )
                wall_assignment[wall] = ap
                remaining.remove(ap)

            long_wall_aps = [
                wall_assignment.get("bottom_mid"),
                wall_assignment.get("top_mid"),
            ]
            long_wall_aps = [x for x in long_wall_aps if x is not None]

            left_short = wall_assignment.get("left_mid")
            right_short = wall_assignment.get("right_mid")

            if left_short is not None:
                scenarios["High_left_short"] = google + long_wall_aps + [left_short]
            if right_short is not None:
                scenarios["High_right_short"] = google + long_wall_aps + [right_short]

    # Exploratory scenario containing every AP common to both days.
    if common_aps:
        scenarios["All_common_exploratory"] = common_aps.copy()

    if not scenarios:
        raise ValueError(
            "No valid scenarios could be created. "
            f"Common RTT+RSS APs across both days: {common_aps}"
        )

    return {
        name: list(dict.fromkeys(aps))
        for name, aps in scenarios.items()
    }

def scenario_table(scenarios: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, aps in scenarios.items():
        rows.append({
            "scenario": name,
            "n_physical_aps": len(aps),
            "physical_aps": ";".join(aps),
            "n_features": 2 * len(aps),
            "note": (
                "Unambiguous 4-Google scenario"
                if name == "Medium_4Google"
                else "Topology-defined variant; do not cherry-pick by accuracy"
                if name.startswith("Low") or name.startswith("High")
                else "Exploratory"
            ),
        })
    return pd.DataFrame(rows)


# =============================================================================
# FEATURES
# =============================================================================

def make_X_y(df: pd.DataFrame, aps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = []
    for ap in aps:
        feature_cols.extend([f"RTT_{ap}", f"RSS_{ap}"])

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[["x", "y"]].to_numpy(dtype=float)

    return X, y


# =============================================================================
# PROTOCOL 1: STRATIFIED ROW CV
# =============================================================================

def run_stratified_row_cv(
    df: pd.DataFrame,
    scenario_name: str,
    aps: list[str],
) -> pd.DataFrame:
    X, y = make_X_y(df, aps)
    rp_labels = rp_label_from_xy(df)

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    rows = []
    dummy = np.zeros(len(df))

    for model_name, mf in [
        ("RF300-F1.0", BASELINE_MAX_FEATURES),
        ("RF300-F0.7", PROPOSED_MAX_FEATURES),
    ]:
        for fold, (tr, va) in enumerate(skf.split(dummy, rp_labels), start=1):
            model = build_rf(mf)
            train_time = timed_fit(model, X[tr], y[tr])
            pred, pred_time = timed_predict(model, X[va])

            rows.append({
                "protocol": "stratified_row_cv",
                "scenario": scenario_name,
                "model": model_name,
                "fold": fold,
                "train_rows": len(tr),
                "validation_rows": len(va),
                "train_rps": len(np.unique(rp_labels[tr])),
                "validation_rps": len(np.unique(rp_labels[va])),
                "train_time_s": train_time,
                "predict_time_s": pred_time,
                **metrics_2d(y[va], pred),
            })

            del model
            gc.collect()

    return pd.DataFrame(rows)


# =============================================================================
# PROTOCOL 2: STRICT RP-GROUPED CV
# =============================================================================

def run_grouped_rp_cv(
    df: pd.DataFrame,
    scenario_name: str,
    aps: list[str],
) -> pd.DataFrame:
    X, y = make_X_y(df, aps)
    groups = make_group_ids(df)

    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(df))

    rows = []

    for model_name, mf in [
        ("RF300-F1.0", BASELINE_MAX_FEATURES),
        ("RF300-F0.7", PROPOSED_MAX_FEATURES),
    ]:
        for fold, (tr, va) in enumerate(gkf.split(dummy, y, groups), start=1):
            model = build_rf(mf)
            train_time = timed_fit(model, X[tr], y[tr])
            pred, pred_time = timed_predict(model, X[va])

            rows.append({
                "protocol": "strict_rp_grouped_cv",
                "scenario": scenario_name,
                "model": model_name,
                "fold": fold,
                "train_rows": len(tr),
                "validation_rows": len(va),
                "train_rps": len(np.unique(groups[tr])),
                "validation_rps": len(np.unique(groups[va])),
                "train_time_s": train_time,
                "predict_time_s": pred_time,
                **metrics_2d(y[va], pred),
            })

            del model
            gc.collect()

    return pd.DataFrame(rows)


# =============================================================================
# PROTOCOL 3: CROSS-DAY
# =============================================================================

def run_cross_day(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_day: int,
    test_day: int,
    scenario_name: str,
    aps: list[str],
) -> pd.DataFrame:
    X_train, y_train = make_X_y(train_df, aps)
    X_test, y_test = make_X_y(test_df, aps)

    rows = []

    for model_name, mf in [
        ("RF300-F1.0", BASELINE_MAX_FEATURES),
        ("RF300-F0.7", PROPOSED_MAX_FEATURES),
    ]:
        model = build_rf(mf)
        train_time = timed_fit(model, X_train, y_train)
        pred, pred_time = timed_predict(model, X_test)

        rows.append({
            "protocol": "cross_day",
            "scenario": scenario_name,
            "model": model_name,
            "train_day": train_day,
            "test_day": test_day,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_rps": train_df[["x", "y"]].drop_duplicates().shape[0],
            "test_rps": test_df[["x", "y"]].drop_duplicates().shape[0],
            "train_time_s": train_time,
            "predict_time_s": pred_time,
            **metrics_2d(y_test, pred),
        })

        del model
        gc.collect()

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIES
# =============================================================================

def summarize_cv(fold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_df
        .groupby(["protocol", "scenario", "model"], as_index=False)
        .agg(
            rmse_2d_m=("rmse_2d_m", "mean"),
            rmse_2d_std_m=("rmse_2d_m", "std"),
            mean_error_m=("mean_error_m", "mean"),
            median_error_m=("median_error_m", "mean"),
            p90_error_m=("p90_error_m", "mean"),
            p95_error_m=("p95_error_m", "mean"),
            train_time_s=("train_time_s", "mean"),
            predict_time_s=("predict_time_s", "mean"),
        )
    )


def paired_improvement_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (protocol, scenario), g in summary.groupby(["protocol", "scenario"]):
        if set(g["model"]) != {"RF300-F1.0", "RF300-F0.7"}:
            continue

        b = g[g["model"] == "RF300-F1.0"].iloc[0]
        p = g[g["model"] == "RF300-F0.7"].iloc[0]

        rows.append({
            "protocol": protocol,
            "scenario": scenario,
            "baseline_rmse_2d_m": b["rmse_2d_m"],
            "proposed_rmse_2d_m": p["rmse_2d_m"],
            "rmse_reduction_pct": pct_reduction(
                p["rmse_2d_m"],
                b["rmse_2d_m"],
            ),
            "baseline_p90_m": b["p90_error_m"],
            "proposed_p90_m": p["p90_error_m"],
            "p90_reduction_pct": pct_reduction(
                p["p90_error_m"],
                b["p90_error_m"],
            ),
            "baseline_train_time_s": b["train_time_s"],
            "proposed_train_time_s": p["train_time_s"],
            "train_time_reduction_pct": pct_reduction(
                p["train_time_s"],
                b["train_time_s"],
            ),
            "baseline_predict_time_s": b["predict_time_s"],
            "proposed_predict_time_s": p["predict_time_s"],
            "predict_time_reduction_pct": pct_reduction(
                p["predict_time_s"],
                b["predict_time_s"],
            ),
        })

    return pd.DataFrame(rows)


def summarize_cross_day(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (scenario, tr_day, te_day), g in df.groupby(
        ["scenario", "train_day", "test_day"]
    ):
        if set(g["model"]) != {"RF300-F1.0", "RF300-F0.7"}:
            continue

        b = g[g["model"] == "RF300-F1.0"].iloc[0]
        p = g[g["model"] == "RF300-F0.7"].iloc[0]

        rows.append({
            "scenario": scenario,
            "train_day": tr_day,
            "test_day": te_day,
            "baseline_rmse_2d_m": b["rmse_2d_m"],
            "proposed_rmse_2d_m": p["rmse_2d_m"],
            "rmse_reduction_pct": pct_reduction(
                p["rmse_2d_m"],
                b["rmse_2d_m"],
            ),
            "baseline_p90_m": b["p90_error_m"],
            "proposed_p90_m": p["p90_error_m"],
            "p90_reduction_pct": pct_reduction(
                p["p90_error_m"],
                b["p90_error_m"],
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# PLOTS
# =============================================================================

def plot_ap_map(ap_locations: pd.DataFrame, combined: pd.DataFrame) -> None:
    rps = combined[["x", "y"]].drop_duplicates()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(rps["x"], rps["y"], s=20, alpha=0.5, label="Reference points")

    for vendor, g in ap_locations.groupby("vendor_group"):
        ax.scatter(
            g["inferred_x_m"],
            g["inferred_y_m"],
            s=90,
            marker="^",
            label=f"{vendor} APs (inferred)",
        )
        for row in g.itertuples():
            ax.annotate(
                row.physical_ap,
                (row.inferred_x_m, row.inferred_y_m),
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("EETAC reference points and inferred physical AP locations")
    ax.legend()
    save_fig(fig, "01_eetac_rp_ap_map")


def plot_protocol_rmse(summary: pd.DataFrame, protocol: str) -> None:
    g = summary[summary["protocol"] == protocol].copy()
    if g.empty:
        return

    scenarios = list(dict.fromkeys(g["scenario"].tolist()))
    x = np.arange(len(scenarios))
    width = 0.36

    base = []
    prop = []

    for s in scenarios:
        sg = g[g["scenario"] == s]
        base.append(float(sg[sg["model"] == "RF300-F1.0"]["rmse_2d_m"].iloc[0]))
        prop.append(float(sg[sg["model"] == "RF300-F0.7"]["rmse_2d_m"].iloc[0]))

    fig, ax = plt.subplots(figsize=(max(9, len(scenarios) * 1.5), 5))
    ax.bar(x - width / 2, base, width, label="RF300-F1.0")
    ax.bar(x + width / 2, prop, width, label="RF300-F0.7")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("2D RMSE (m)")
    ax.set_title(protocol.replace("_", " ").title())
    ax.legend()
    save_fig(fig, f"rmse_{protocol}")


def plot_improvement(improvement: pd.DataFrame, protocol: str) -> None:
    g = improvement[improvement["protocol"] == protocol].copy()
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(g)), 5))
    ax.bar(g["scenario"], g["rmse_reduction_pct"])
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("RMSE reduction of F=0.7 vs F=1.0 (%)")
    ax.set_title(f"Subspace RF improvement: {protocol.replace('_', ' ')}")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, f"improvement_{protocol}")


def plot_cross_day(cross_summary: pd.DataFrame) -> None:
    if cross_summary.empty:
        return

    labels = [
        f"{r.scenario}\nD{r.train_day}->D{r.test_day}"
        for r in cross_summary.itertuples()
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.25), 5))
    ax.bar(
        x - width / 2,
        cross_summary["baseline_rmse_2d_m"],
        width,
        label="RF300-F1.0",
    )
    ax.bar(
        x + width / 2,
        cross_summary["proposed_rmse_2d_m"],
        width,
        label="RF300-F0.7",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("2D RMSE (m)")
    ax.set_title("Cross-day temporal generalization")
    ax.legend()
    save_fig(fig, "cross_day_rmse")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("EETAC EXTERNAL VALIDATION: RF300-F1.0 vs SUBSPACE RF300-F0.7")
    print("=" * 100)

    day_tables = {}
    audits = []

    for day in [1, 2]:
        print(f"\nLoading Day {day}...")
        table, audit = build_day_table(day)
        day_tables[day] = table
        audits.append(audit)

        n_rp = table[["x", "y"]].drop_duplicates().shape[0]
        print(
            f"Day {day}: {len(table)} rows, {n_rp} RPs, "
            f"{len(PHYSICAL_APS)} physical AP groups."
        )

    # Keep only physical APs available in BOTH RTT and RSS on BOTH days.
    day_available = {}
    for day in [1, 2]:
        day_available[day] = {
            ap for ap in PHYSICAL_APS
            if f"RTT_{ap}" in day_tables[day].columns
            and f"RSS_{ap}" in day_tables[day].columns
        }

    common_aps = [
        ap for ap in PHYSICAL_APS
        if ap in day_available[1] and ap in day_available[2]
    ]

    # Day 2 is recorded in a y-mirrored frame relative to Day 1.  Detect it
    # from the data and align before any pooling or cross-day evaluation.
    frame_audit = detect_day_frame(day_tables[1], day_tables[2])
    frame_audit.to_csv(OUT_DIR / "day_frame_alignment_audit.csv", index=False)
    print("\nDay-frame alignment audit:")
    print(frame_audit.to_string(index=False))
    if ALIGN_DAY_FRAMES and frame_audit["selected"].iloc[0] == "mirror_y":
        day_tables[2]["y"] = (CANONICAL_Y.max() - day_tables[2]["y"]).round(6)
        print("Applied Day-2 y -> 8 - y alignment.")

    print(f"\nDay 1 available physical APs: {sorted(day_available[1])}")
    print(f"Day 2 available physical APs: {sorted(day_available[2])}")
    print(f"Common RTT+RSS APs: {common_aps}")

    if not common_aps:
        raise ValueError("No physical AP has both RTT and RSS on both days.")

    # Combine only after determining common features.
    combined = pd.concat(
        [day_tables[1], day_tables[2]],
        ignore_index=True,
        sort=False,
    )

    combined_rps = combined[["x", "y"]].drop_duplicates().shape[0]
    print(f"Combined canonical reference points: {combined_rps}")
    if combined_rps != 25:
        print(
            "WARNING: Expected 25 shared physical RPs after coordinate "
            f"canonicalization, but found {combined_rps}. Inspect coordinates."
        )

    # Dataset summary.
    dataset_summary = pd.DataFrame([
        {
            "day": day,
            "rows": len(day_tables[day]),
            "reference_points": day_tables[day][["x", "y"]].drop_duplicates().shape[0],
            "samples_per_rp_min":
                int(day_tables[day].groupby(["x", "y"]).size().min()),
            "samples_per_rp_max":
                int(day_tables[day].groupby(["x", "y"]).size().max()),
            "x_min_m": float(day_tables[day]["x"].min()),
            "x_max_m": float(day_tables[day]["x"].max()),
            "y_min_m": float(day_tables[day]["y"].min()),
            "y_max_m": float(day_tables[day]["y"].max()),
        }
        for day in [1, 2]
    ])

    dataset_summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(
        OUT_DIR / "ap_bssid_audit.csv",
        index=False,
    )

    # Infer physical AP locations using both days.
    # Only common APs are relevant for cross-day comparison.
    common_cols = META_COLS + ["day"]
    for ap in common_aps:
        common_cols.extend([f"RTT_{ap}", f"RSS_{ap}"])
    common_cols = [c for c in common_cols if c in combined.columns]
    combined_common = combined[common_cols].copy()

    ap_locations = infer_ap_locations(combined_common)
    ap_locations.to_csv(
        OUT_DIR / "physical_ap_inferred_locations.csv",
        index=False,
    )

    scenarios = build_scenarios(ap_locations, common_aps)
    scenario_df = scenario_table(scenarios)
    scenario_df.to_csv(
        OUT_DIR / "scenario_definition.csv",
        index=False,
    )

    print("\nInferred AP locations:")
    print(ap_locations.to_string(index=False))

    print("\nScenarios:")
    print(scenario_df.to_string(index=False))

    plot_ap_map(ap_locations, combined_common)

    # -------------------------------------------------------------------------
    # Evaluate CV protocols on combined Day1 + Day2.
    # -------------------------------------------------------------------------
    fold_frames = []

    for scenario_name, aps in scenarios.items():
        print(f"\n[{scenario_name}] APs={aps}")

        print("  - Stratified row CV")
        fold_frames.append(
            run_stratified_row_cv(
                combined_common,
                scenario_name,
                aps,
            )
        )

        print("  - Strict RP-grouped CV")
        fold_frames.append(
            run_grouped_rp_cv(
                combined_common,
                scenario_name,
                aps,
            )
        )

    fold_results = pd.concat(fold_frames, ignore_index=True)
    fold_results.to_csv(
        OUT_DIR / "fold_results.csv",
        index=False,
    )

    summary = summarize_cv(fold_results)
    summary.to_csv(
        OUT_DIR / "summary_results.csv",
        index=False,
    )

    improvement = paired_improvement_summary(summary)
    improvement.to_csv(
        OUT_DIR / "paired_improvement_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Cross-day temporal generalization.
    # -------------------------------------------------------------------------
    cross_frames = []

    for scenario_name, aps in scenarios.items():
        print(f"\nCross-day: {scenario_name}")

        cross_frames.append(
            run_cross_day(
                day_tables[1],
                day_tables[2],
                1,
                2,
                scenario_name,
                aps,
            )
        )

        cross_frames.append(
            run_cross_day(
                day_tables[2],
                day_tables[1],
                2,
                1,
                scenario_name,
                aps,
            )
        )

    cross_day = pd.concat(cross_frames, ignore_index=True)
    cross_day.to_csv(
        OUT_DIR / "cross_day_results.csv",
        index=False,
    )

    cross_summary = summarize_cross_day(cross_day)
    cross_summary.to_csv(
        OUT_DIR / "cross_day_improvement_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Plots.
    # -------------------------------------------------------------------------
    for protocol in [
        "stratified_row_cv",
        "strict_rp_grouped_cv",
    ]:
        plot_protocol_rmse(summary, protocol)
        plot_improvement(improvement, protocol)

    plot_cross_day(cross_summary)

    # -------------------------------------------------------------------------
    # Console result: prioritize the unambiguous Medium scenario.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PRIMARY EXTERNAL-VALIDATION RESULT: MEDIUM (4 GOOGLE APs)")
    print("=" * 100)

    med_summary = summary[summary["scenario"] == "Medium_4Google"]
    print(
        med_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nMedium improvement:")
    med_imp = improvement[improvement["scenario"] == "Medium_4Google"]
    print(
        med_imp.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nMedium cross-day:")
    med_cross = cross_summary[cross_summary["scenario"] == "Medium_4Google"]
    print(
        med_cross.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 100)
    print("INTERPRETATION RULES")
    print("=" * 100)
    print(
        "1. Use Medium_4Google as the cleanest external benchmark first.\n"
        "2. Report BOTH Low diagonal variants until the exact paper AP labels are confirmed.\n"
        "3. Report BOTH High short-wall variants until the exact Linksys subset is confirmed.\n"
        "4. Do not tune max_features on this dataset: F=0.7 was frozen from the original study.\n"
        "5. Primary accuracy metric here is TRUE 2D RMSE in metres.\n"
        "6. Stratified-row CV and RP-grouped CV answer different research questions; never mix them.\n"
        "7. Cross-day results test temporal robustness and are especially useful for the paper.\n"
        "8. The script snaps Day-1 y=8.1 m to the canonical y=8.0 m top row so the same physical RP is grouped consistently."
    )

    print(f"\nResults saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
