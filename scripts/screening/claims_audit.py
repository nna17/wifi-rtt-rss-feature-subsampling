"""
claims_audit.py
===============

Independent audit of the main claims of the random-subspace RF paper.

Run from the project root:
    python scripts/screening/claims_audit.py [--quick]

Writes CSVs to audit_results/.

Part A  max_features sweep x seeds on the original benchmark
        (official RP-disjoint test + 5-fold RP-grouped CV on TRAIN).
        Question: is 0.7 special, and is F0.7 - F1.0 larger than seed noise?
Part B  stronger / alternative baselines on the original benchmark
        (ExtraTrees, KNN, HistGradientBoosting, RF with sqrt / one-third).
Part C  EETAC with and without the Day-2 y-axis correction (y -> 8 - y),
        all three protocols, F1.0 vs F0.7, several seeds.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts" / "original_dataset"))
sys.path.insert(0, str(ROOT / "scripts" / "eetac"))

import paper_all_contributions_wifi_rtt_rss as O  # noqa: E402
import eetac_external_validation_subspace_rf_v2 as E  # noqa: E402

OUT = ROOT / "audit_results"
OUT.mkdir(exist_ok=True)


def rf(max_features, seed, n_trees=300):
    return RandomForestRegressor(
        n_estimators=n_trees, max_features=max_features, min_samples_leaf=1,
        bootstrap=True, n_jobs=-1, random_state=seed,
    )


def metrics(y, p, grid):
    r = (y - p) * grid
    rad = np.linalg.norm(r, axis=1)
    return {
        "paper_metric_m": float(np.mean(np.sqrt(np.mean(r ** 2, axis=0)))),
        "rmse2d_m": float(np.sqrt(np.mean(rad ** 2))),
        "mean_m": float(rad.mean()),
        "p90_m": float(np.quantile(rad, 0.90)),
    }


def grouped_oof(make_model, X, y, groups, n_splits=5):
    pred = np.zeros_like(y, dtype=float)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
    return pred


def load_original(name):
    tr, te = O.load_dataset(name)
    rtt, rss = O.detect_feature_columns(tr)
    Xtr, ytr, _ = O.extract_features_target(tr, rtt, rss)
    Xte, yte, _ = O.extract_features_target(te, rtt, rss)
    return Xtr, ytr, Xte, yte, O.make_rp_groups(tr), O.make_rp_groups(te), O.GRID_SIZE_M[name]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# PART A
# -----------------------------------------------------------------------------

def part_a(seeds, datasets):
    rows = []
    for name in datasets:
        Xtr, ytr, Xte, yte, gtr, _, grid = load_original(name)
        p = Xtr.shape[1]
        if p <= 8:
            mtrys = list(range(1, p + 1))
        else:
            mtrys = sorted({1, 2, int(np.sqrt(p)), p // 3, p // 2, int(0.7 * p), int(0.85 * p), p})
        for mtry in mtrys:
            for seed in seeds:
                t0 = time.perf_counter()
                m = rf(mtry, seed).fit(Xtr, ytr)
                test = metrics(yte, m.predict(Xte), grid)
                oof = metrics(ytr, grouped_oof(lambda: rf(mtry, seed), Xtr, ytr, gtr), grid)
                rows.append({
                    "dataset": name, "p": p, "mtry": mtry, "ratio": mtry / p, "seed": seed,
                    **{f"test_{k}": v for k, v in test.items()},
                    **{f"cv_{k}": v for k, v in oof.items()},
                })
                log(f"A {name} mtry={mtry}/{p} seed={seed} test={test['paper_metric_m']:.4f} "
                    f"cv={oof['paper_metric_m']:.4f} ({time.perf_counter() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(OUT / "A_mtry_sweep.csv", index=False)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# PART B
# -----------------------------------------------------------------------------

def baselines(p):
    return {
        "RF-F1.0": lambda: rf(1.0, 0),
        "RF-F0.7": lambda: rf(0.7, 0),
        "RF-sqrt": lambda: rf("sqrt", 0),
        "RF-third": lambda: rf(max(1, p // 3), 0),
        "ET-F1.0": lambda: ExtraTreesRegressor(300, max_features=1.0, n_jobs=-1, random_state=0),
        "ET-F0.7": lambda: ExtraTreesRegressor(300, max_features=0.7, n_jobs=-1, random_state=0),
        "KNN-k10-dist": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance", n_jobs=-1)),
        "HGB": lambda: MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0)),
    }


def part_b(datasets):
    rows = []
    for name in datasets:
        Xtr, ytr, Xte, yte, gtr, _, grid = load_original(name)
        for model_name, make in baselines(Xtr.shape[1]).items():
            t0 = time.perf_counter()
            test = metrics(yte, make().fit(Xtr, ytr).predict(Xte), grid)
            oof = metrics(ytr, grouped_oof(make, Xtr, ytr, gtr), grid)
            rows.append({"dataset": name, "model": model_name,
                         **{f"test_{k}": v for k, v in test.items()},
                         **{f"cv_{k}": v for k, v in oof.items()}})
            log(f"B {name} {model_name} test={test['paper_metric_m']:.4f} cv={oof['paper_metric_m']:.4f} "
                f"({time.perf_counter() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(OUT / "B_baselines.csv", index=False)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# PART C
# -----------------------------------------------------------------------------

def part_c(seeds):
    t1, _ = E.build_day_table(1)
    t2, _ = E.build_day_table(2)
    scen = {"Medium_4Google": ["G1", "G2", "G3", "G4"],
            "High_4Google_3Linksys": ["G1", "G2", "G3", "G4", "L1", "L2", "L3"]}
    rows = []
    for flip in (False, True):
        d2 = t2.copy()
        if flip:
            d2["y"] = 8.0 - d2["y"]
        comb = pd.concat([t1, d2], ignore_index=True)
        groups = E.make_group_ids(comb)
        rp = E.rp_label_from_xy(comb)
        for sname, aps in scen.items():
            X, y = E.make_X_y(comb, aps)
            for seed in seeds:
                for mf in (1.0, 0.7):
                    rec = {"y_flip_day2": flip, "scenario": sname, "seed": seed, "max_features": mf}
                    # sample-level (stratified by RP), pooled OOF RMSE
                    pred = np.zeros_like(y)
                    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, rp):
                        pred[va] = rf(mf, seed).fit(X[tr], y[tr]).predict(X[va])
                    rec["sample_cv_rmse2d"] = metrics(y, pred, 1.0)["rmse2d_m"]
                    pred = grouped_oof(lambda: rf(mf, seed), X, y, groups)
                    rec["rp_grouped_cv_rmse2d"] = metrics(y, pred, 1.0)["rmse2d_m"]
                    for a, b, lab in ((t1, d2, "d1_to_d2"), (d2, t1, "d2_to_d1")):
                        Xa, ya = E.make_X_y(a, aps)
                        Xb, yb = E.make_X_y(b, aps)
                        rec[f"{lab}_rmse2d"] = metrics(yb, rf(mf, seed).fit(Xa, ya).predict(Xb), 1.0)["rmse2d_m"]
                    rows.append(rec)
                    log(f"C flip={flip} {sname} seed={seed} F{mf}: " +
                        " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k.endswith("rmse2d")))
            pd.DataFrame(rows).to_csv(OUT / "C_eetac_flip.csv", index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="office/apartment only, 2 seeds")
    ap.add_argument("--parts", default="ABC")
    a = ap.parse_args()
    seeds = [0, 1] if a.quick else [0, 1, 2]
    datasets = ["office", "apartment"] if a.quick else ["office", "apartment", "building"]
    if "C" in a.parts:
        part_c(seeds)
    if "B" in a.parts:
        part_b(datasets)
    if "A" in a.parts:
        part_a(seeds, datasets)
    log("done")


if __name__ == "__main__":
    main()
