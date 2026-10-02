"""
paper_extensions.py
===================

Extension experiments of the paper.  Run from the project root:

    python scripts/extensions/paper_extensions.py --parts E1,E2,E3,E3b,E5,E6,E7,E8,E9,E11

Outputs -> audit_results/extensions/

E1  Radio-map density x subspace ratio (10 random RP subsets, 3 environments)
E2  How should the ratio be chosen?  OOB / sample-CV / RP-grouped-CV /
    leave-one-environment-out minimax regret / fixed defaults
E3  Literature alternatives: Breiman p/3, ExtraTrees, Rotation Forest, WKNN
E3b Gradient-boosted trees under the E3 protocol
E4  Calibration with correlated scans: scan-level vs RP-count split
    conformal, global vs tree-spread-normalised scores (superseded by E8)
E5  EETAC (frame-aligned) ratio sweep: sample CV, RP-grouped CV, cross-day
E6  Compact forests: accuracy vs serialized size and inference time
E7  Radio-map density x subspace ratio for extremely randomized trees
E8  Calibration rules on the same splits: scan-level, group (one scan per
    RP, valid), RP-count (conservative heuristic)
E9  rho=0.7 vs rho=1 on the official tests, 10 seeds
E11 rho=0.7 vs rho=1 on all EETAC protocols, 10 seeds, RP-clustered bootstrap
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts" / "original_dataset"))
sys.path.insert(0, str(ROOT / "scripts" / "eetac"))
import paper_all_contributions_wifi_rtt_rss as O  # noqa: E402
import eetac_external_validation_subspace_rf_v2 as E  # noqa: E402

OUT = ROOT / "audit_results" / "extensions"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ("office", "apartment", "building")
RATIOS = (0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
SEEDS = (0, 1, 2)
# Random RP subsets for the density sweeps (E1, E7); the first three draws are
# identical to the original three-seed runs.
DENSITY_SEEDS = tuple(range(10))
# Seeds for the multi-seed headline comparisons (E9, E11).
SEEDS10 = tuple(range(10))
BOOT = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mtry_of(ratio, p):
    # Same rule as scikit-learn for a float max_features and as Eq. (9) of the
    # paper: m_try = max(1, floor(rho * p)).
    return max(1, int(np.floor(ratio * p + 1e-9)))


def rf(mtry, seed=0, **kw):
    return RandomForestRegressor(n_estimators=300, max_features=mtry, n_jobs=-1, random_state=seed, **kw)


def metrics(y, p, grid=1.0):
    r = (y - p) * grid
    rad = np.linalg.norm(r, axis=1)
    return {
        "paper_metric_m": float(np.mean(np.sqrt(np.mean(r ** 2, axis=0)))),
        "rmse2d_m": float(np.sqrt(np.mean(rad ** 2))),
        "p90_m": float(np.quantile(rad, 0.9)),
    }


def load(name):
    tr, te = O.load_dataset(name)
    rtt, rss = O.detect_feature_columns(tr)
    Xtr, ytr, _ = O.extract_features_target(tr, rtt, rss)
    Xte, yte, _ = O.extract_features_target(te, rtt, rss)
    return Xtr, ytr, Xte, yte, O.make_rp_groups(tr), O.make_rp_groups(te), O.GRID_SIZE_M[name]


def oof(make, X, y, splits):
    pred = np.zeros_like(y, dtype=float)
    for tr, va in splits:
        pred[va] = make().fit(X[tr], y[tr]).predict(X[va])
    return pred


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


# -----------------------------------------------------------------------------
# E1
# -----------------------------------------------------------------------------

def e1():
    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, g, _, grid = load(name)
        p, ug = Xtr.shape[1], np.unique(g)
        for frac in (1.0, 0.75, 0.5, 0.25):
            for seed in DENSITY_SEEDS:
                rs = np.random.default_rng(1000 + seed)
                keep = ug if frac == 1.0 else rs.choice(ug, int(round(frac * len(ug))), replace=False)
                m = np.isin(g, keep)
                for ratio in RATIOS:
                    pred = rf(mtry_of(ratio, p), seed).fit(Xtr[m], ytr[m]).predict(Xte)
                    rows.append({"dataset": name, "rp_fraction": frac, "train_rps": len(keep),
                                 "seed": seed, "ratio": ratio, **metrics(yte, pred, grid)})
                log(f"E1 {name} frac={frac} seed={seed}")
                save(rows, "E1_density_ratio.csv")


# -----------------------------------------------------------------------------
# E2
# -----------------------------------------------------------------------------

def e2():
    """Per ratio: OOB (sample-level), 5-fold sample CV, 5-fold RP-grouped CV,
    and official test.  Selection rules are evaluated afterwards in summarise()."""
    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, g, _, grid = load(name)
        p = Xtr.shape[1]
        sample_splits = list(KFold(5, shuffle=True, random_state=0).split(Xtr))
        group_splits = list(GroupKFold(5).split(Xtr, ytr, g))
        for ratio in RATIOS:
            mt = mtry_of(ratio, p)
            full = rf(mt, 0, oob_score=True).fit(Xtr, ytr)
            rec = {"dataset": name, "ratio": ratio, "mtry": mt,
                   "oob_paper_metric_m": metrics(ytr, full.oob_prediction_, grid)["paper_metric_m"],
                   "test_paper_metric_m": metrics(yte, full.predict(Xte), grid)["paper_metric_m"]}
            rec["sample_cv_paper_metric_m"] = metrics(
                ytr, oof(lambda: rf(mt, 0), Xtr, ytr, sample_splits), grid)["paper_metric_m"]
            rec["group_cv_paper_metric_m"] = metrics(
                ytr, oof(lambda: rf(mt, 0), Xtr, ytr, group_splits), grid)["paper_metric_m"]
            rows.append(rec)
            log(f"E2 {name} ratio={ratio} " + " ".join(f"{k[:-15]}={v:.4f}" for k, v in rec.items()
                                                       if k.endswith("paper_metric_m")))
            save(rows, "E2_selection_curves.csv")


# -----------------------------------------------------------------------------
# E3
# -----------------------------------------------------------------------------

def _rotation_tree(X, y, seed, subset_size=3):
    rs = np.random.default_rng(seed)
    p = X.shape[1]
    perm = rs.permutation(p)
    subsets = [perm[i:i + subset_size] for i in range(0, p, subset_size)]
    R = np.zeros((p, p))
    for s in subsets:
        idx = rs.choice(len(X), size=max(len(s) + 1, int(0.75 * len(X))), replace=False)
        pca = PCA(random_state=seed).fit(X[np.ix_(idx, s)])
        comps = pca.components_
        R[np.ix_(s, s[: comps.shape[0]])] = comps.T
    boot = rs.integers(0, len(X), len(X))
    tree = DecisionTreeRegressor(random_state=seed).fit(X[boot] @ R, y[boot])
    return tree, R


class RotationForest:
    """Rotation Forest (Rodriguez, Kuncheva & Alonso, 2006) for regression.

    Features are standardised, split into random disjoint subsets of 3, each
    subset is rotated by PCA fitted on a 75% subsample, and a full tree is
    grown on the rotated, bootstrapped data.
    """

    def __init__(self, n_estimators=300, seed=0):
        self.n, self.seed = n_estimators, seed

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.members = Parallel(n_jobs=-1)(
            delayed(_rotation_tree)(Z, y, self.seed * 100000 + t) for t in range(self.n))
        return self

    def predict(self, X):
        Z = self.sc.transform(X)
        return np.mean([t.predict(Z @ R) for t, R in self.members], axis=0)


def literature_models(p, seed):
    return {
        "RF-F1.0 (bagging)": lambda: rf(p, seed),
        "RF-F0.7": lambda: rf(mtry_of(0.7, p), seed),
        "RF-p/3 (Breiman)": lambda: rf(max(1, p // 3), seed),
        "ExtraTrees-F1.0": lambda: ExtraTreesRegressor(300, max_features=1.0, n_jobs=-1, random_state=seed),
        "RotationForest": lambda: RotationForest(300, seed),
        "WKNN-k5": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(5, weights="distance")),
    }


def e3():
    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, g, _, grid = load(name)
        group_splits = list(GroupKFold(5).split(Xtr, ytr, g))
        seeds = SEEDS if name != "building" else (0,)
        for seed in seeds:
            for mname, make in literature_models(Xtr.shape[1], seed).items():
                t0 = time.perf_counter()
                test = metrics(yte, make().fit(Xtr, ytr).predict(Xte), grid)
                cv = metrics(ytr, oof(make, Xtr, ytr, group_splits), grid)
                rows.append({"dataset": name, "seed": seed, "model": mname,
                             **{f"test_{k}": v for k, v in test.items()},
                             **{f"cv_{k}": v for k, v in cv.items()}})
                log(f"E3 {name} s{seed} {mname} test={test['paper_metric_m']:.4f} "
                    f"cv={cv['paper_metric_m']:.4f} ({time.perf_counter() - t0:.0f}s)")
                save(rows, "E3_literature.csv")


# -----------------------------------------------------------------------------
# E4
# -----------------------------------------------------------------------------

def q_level(n, conf):
    return min(1.0, math.ceil((n + 1) * conf) / n)


def tree_scale(model, X, grid):
    P = np.stack([t.predict(X) for t in model.estimators_]) * grid
    return np.sqrt(P.var(axis=0, ddof=1).sum(axis=1) / 2.0)


def e4(n_splits=20):
    """Scan-level split conformal treats ~120 correlated scans per RP as
    exchangeable draws; the finite-sample correction then uses n = #scans.
    The RP-corrected variant pools scores but applies the correction with
    n = #calibration RPs (the effective number of exchangeable units), and
    the RP-max variant uses one score per RP (its within-RP quantile), giving
    a group-level guarantee."""
    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, g, gte, grid = load(name)
        p = Xtr.shape[1]
        n_sp = n_splits if name != "building" else 10
        for sp in range(n_sp):
            fit, cal = next(GroupShuffleSplit(1, test_size=0.25, random_state=sp).split(Xtr, ytr, g))
            m = rf(mtry_of(0.7, p), sp).fit(Xtr[fit], ytr[fit])
            r_cal = np.linalg.norm((ytr[cal] - m.predict(Xtr[cal])) * grid, axis=1)
            r_te = np.linalg.norm((yte - m.predict(Xte)) * grid, axis=1)
            s_cal, s_te = tree_scale(m, Xtr[cal], grid), tree_scale(m, Xte, grid)
            floor = float(np.median(s_cal))
            G = len(np.unique(g[cal]))
            for score_name, sc, st in (("circle", r_cal, r_te),
                                       ("normalised", r_cal / (s_cal + floor), r_te / (s_te + floor))):
                for conf in (0.90, 0.95):
                    rules = {
                        "scan-level": np.quantile(sc, q_level(len(sc), conf), method="higher"),
                        "RP-corrected": np.quantile(sc, q_level(G, conf), method="higher"),
                    }
                    for rule, q in rules.items():
                        cov = st <= q
                        radius = q if score_name == "circle" else q * (s_te + floor)
                        per_rp = pd.Series(cov).groupby(gte).mean()
                        rows.append({
                            "dataset": name, "split": sp, "cal_rps": G, "score": score_name,
                            "rule": rule, "confidence": conf,
                            "coverage": float(cov.mean()),
                            "rp_coverage_min": float(per_rp.min()),
                            "mean_area_m2": float(np.mean(math.pi * np.asarray(radius) ** 2)),
                        })
            log(f"E4 {name} split={sp} G={G}")
            save(rows, "E4_calibration.csv")


# -----------------------------------------------------------------------------
# E5
# -----------------------------------------------------------------------------

def eetac_tables():
    t1, _ = E.build_day_table(1)
    t2, _ = E.build_day_table(2)
    if E.detect_day_frame(t1, t2)["selected"].iloc[0] == "mirror_y":
        t2["y"] = (E.CANONICAL_Y.max() - t2["y"]).round(6)
    return t1, t2


def e5():
    t1, t2 = eetac_tables()
    comb = pd.concat([t1, t2], ignore_index=True)
    groups = E.make_group_ids(comb)
    rp = E.rp_label_from_xy(comb)
    scen = {"Medium_4Google": ["G1", "G2", "G3", "G4"],
            "High_4Google_3Linksys": ["G1", "G2", "G3", "G4", "L1", "L2", "L3"]}
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for sname, aps in scen.items():
        X, y = E.make_X_y(comb, aps)
        p = X.shape[1]
        for seed in SEEDS:
            ss = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, rp))
            gs = list(GroupKFold(5).split(X, y, groups))
            for ratio in RATIOS:
                mt = mtry_of(ratio, p)
                rec = {"scenario": sname, "p": p, "seed": seed, "ratio": ratio, "mtry": mt,
                       "sample_cv": metrics(y, oof(lambda: rf(mt, seed), X, y, ss))["rmse2d_m"],
                       "rp_grouped_cv": metrics(y, oof(lambda: rf(mt, seed), X, y, gs))["rmse2d_m"]}
                for a, b, lab in ((t1, t2, "d1_to_d2"), (t2, t1, "d2_to_d1")):
                    Xa, ya = E.make_X_y(a, aps)
                    Xb, yb = E.make_X_y(b, aps)
                    rec[lab] = metrics(yb, rf(mt, seed).fit(Xa, ya).predict(Xb))["rmse2d_m"]
                rows.append(rec)
            log(f"E5 {sname} seed={seed}")
            save(rows, "E5_eetac_ratio.csv")


# -----------------------------------------------------------------------------
# E6
# -----------------------------------------------------------------------------

def e6():
    """Model compactness: accuracy vs serialized size and inference time for
    smaller forests (fewer trees / larger leaves / bounded depth).  Inference
    is timed single-threaded, the setting relevant to a phone or edge device."""
    import pickle

    rows = []
    grid_cfg = [(n, leaf, depth) for n in (300, 100, 50) for leaf in (1, 5, 20) for depth in (None, 16)]
    for name in DATASETS:
        Xtr, ytr, Xte, yte, _, _, grid = load(name)
        p = Xtr.shape[1]
        for ratio in (1.0, 0.7, 0.5):
            for n, leaf, depth in grid_cfg:
                m = RandomForestRegressor(n_estimators=n, max_features=mtry_of(ratio, p),
                                          min_samples_leaf=leaf, max_depth=depth,
                                          n_jobs=-1, random_state=0)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr)
                train_s = time.perf_counter() - t0
                m.set_params(n_jobs=1)
                m.predict(Xte[:64])
                t0 = time.perf_counter()
                pred = m.predict(Xte)
                infer_us = (time.perf_counter() - t0) / len(Xte) * 1e6
                rows.append({
                    "dataset": name, "ratio": ratio, "n_trees": n, "min_samples_leaf": leaf,
                    "max_depth": depth if depth else "None",
                    "size_mb": len(pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL)) / 2 ** 20,
                    "total_nodes": int(sum(t.tree_.node_count for t in m.estimators_)),
                    "train_s": train_s, "infer_us_per_scan_1thread": infer_us,
                    **metrics(yte, pred, grid),
                })
                log(f"E6 {name} r={ratio} n={n} leaf={leaf} depth={depth} "
                    f"size={rows[-1]['size_mb']:.1f}MB test={rows[-1]['paper_metric_m']:.4f}")
                save(rows, "E6_model_size.csv")


# -----------------------------------------------------------------------------
# E3b: gradient boosting baseline under the E3 protocol
# -----------------------------------------------------------------------------

def e3b():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, g, _, grid = load(name)
        group_splits = list(GroupKFold(5).split(Xtr, ytr, g))
        seeds = SEEDS if name != "building" else (0,)
        for seed in seeds:
            def make():
                return MultiOutputRegressor(HistGradientBoostingRegressor(
                    max_iter=500, learning_rate=0.05, random_state=seed))
            t0 = time.perf_counter()
            test = metrics(yte, make().fit(Xtr, ytr).predict(Xte), grid)
            cv = metrics(ytr, oof(make, Xtr, ytr, group_splits), grid)
            rows.append({"dataset": name, "seed": seed, "model": "HistGradientBoosting",
                         **{f"test_{k}": v for k, v in test.items()},
                         **{f"cv_{k}": v for k, v in cv.items()}})
            log(f"E3b {name} s{seed} HGB test={test['paper_metric_m']:.4f} cv={cv['paper_metric_m']:.4f} "
                f"({time.perf_counter() - t0:.0f}s)")
            save(rows, "E3b_gradient_boosting.csv")


# -----------------------------------------------------------------------------
# E7: density x ratio for extremely randomized trees
# -----------------------------------------------------------------------------

def e7():
    """Same thinning protocol as E1, with ExtraTreesRegressor instead of RF, to
    test whether the density dependence of the optimal ratio is specific to RF
    or a general property of randomized tree ensembles."""
    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, g, _, grid = load(name)
        p, ug = Xtr.shape[1], np.unique(g)
        for frac in (1.0, 0.75, 0.5, 0.25):
            for seed in DENSITY_SEEDS:
                rs = np.random.default_rng(1000 + seed)
                keep = ug if frac == 1.0 else rs.choice(ug, int(round(frac * len(ug))), replace=False)
                m = np.isin(g, keep)
                for ratio in RATIOS:
                    et = ExtraTreesRegressor(n_estimators=300, max_features=mtry_of(ratio, p),
                                             n_jobs=-1, random_state=seed)
                    pred = et.fit(Xtr[m], ytr[m]).predict(Xte)
                    rows.append({"dataset": name, "rp_fraction": frac, "train_rps": len(keep),
                                 "seed": seed, "ratio": ratio, **metrics(yte, pred, grid)})
                log(f"E7 {name} frac={frac} seed={seed}")
                save(rows, "E7_density_ratio_extratrees.csv")


# -----------------------------------------------------------------------------
# E8: valid group (RP-level) split conformal vs scan-level and RP-count rules
# -----------------------------------------------------------------------------

def e8(n_splits=20):
    """Same RP-level fit/calibration splits and models as E4 (circle score).

    Rules compared:
      scan-level      : pooled scores, finite-sample level with n = #scans
      RP-count        : pooled scores, level with n = G (conservative heuristic)
      group-subsample : one randomly chosen scan per calibration RP, ordinary
                        split conformal on these G exchangeable scores
                        (Dunn, Wasserman & Ramdas, 2023).  Valid marginal
                        coverage for a scan of a new RP when RPs are
                        exchangeable; infinite region if G < 1/alpha - 1.
    """
    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, g, gte, grid = load(name)
        p = Xtr.shape[1]
        n_sp = n_splits if name != "building" else 10
        for sp in range(n_sp):
            fit, cal = next(GroupShuffleSplit(1, test_size=0.25, random_state=sp).split(Xtr, ytr, g))
            m = rf(mtry_of(0.7, p), sp).fit(Xtr[fit], ytr[fit])
            r_cal = np.linalg.norm((ytr[cal] - m.predict(Xtr[cal])) * grid, axis=1)
            r_te = np.linalg.norm((yte - m.predict(Xte)) * grid, axis=1)
            g_cal = g[cal]
            units = np.unique(g_cal)
            G = len(units)
            rs = np.random.default_rng(10_000 + sp)
            one_per_rp = np.array([r_cal[rs.choice(np.flatnonzero(g_cal == u))] for u in units])
            for conf in (0.90, 0.95):
                k = math.ceil((G + 1) * conf)
                q_group = float(np.sort(one_per_rp)[k - 1]) if k <= G else np.inf
                rules = {
                    "scan-level": np.quantile(r_cal, q_level(len(r_cal), conf), method="higher"),
                    "RP-count": np.quantile(r_cal, q_level(G, conf), method="higher"),
                    "group-subsample": q_group,
                }
                for rule, q in rules.items():
                    cov = r_te <= q
                    rows.append({"dataset": name, "split": sp, "cal_rps": G, "rule": rule,
                                 "confidence": conf, "q_m": float(q),
                                 "coverage": float(cov.mean()),
                                 "area_m2": float(math.pi * q ** 2) if np.isfinite(q) else np.inf})
            log(f"E8 {name} split={sp} G={G}")
            save(rows, "E8_group_conformal.csv")


# -----------------------------------------------------------------------------
# E9: ten seeds for the headline official-test comparison
# -----------------------------------------------------------------------------

def e9():
    rows = []
    for name in DATASETS:
        Xtr, ytr, Xte, yte, _, _, grid = load(name)
        p = Xtr.shape[1]
        for seed in SEEDS10:
            for ratio in (1.0, 0.7):
                pred = rf(mtry_of(ratio, p), seed).fit(Xtr, ytr).predict(Xte)
                rows.append({"dataset": name, "seed": seed, "ratio": ratio, **metrics(yte, pred, grid)})
            log(f"E9 {name} seed={seed}")
            save(rows, "E9_multiseed_official.csv")


# -----------------------------------------------------------------------------
# E11: EETAC with ten seeds and RP-clustered bootstrap intervals
# -----------------------------------------------------------------------------

def rp_bootstrap_rmse_diff(y, pa, pb, groups, seed=0, n_boot=BOOT):
    """Paired RP-clustered bootstrap of RMSE(pb) - RMSE(pa) (pa: baseline)."""
    ea = np.sum((y - pa) ** 2, axis=1)
    eb = np.sum((y - pb) ** 2, axis=1)
    units, inv = np.unique(groups, return_inverse=True)
    sa = np.bincount(inv, ea)
    sb = np.bincount(inv, eb)
    n = np.bincount(inv)
    rs = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rs.integers(0, len(units), len(units))
        nn = n[idx].sum()
        diffs[b] = np.sqrt(sb[idx].sum() / nn) - np.sqrt(sa[idx].sum() / nn)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def e11():
    from sklearn.model_selection import StratifiedKFold
    t1, t2 = eetac_tables()
    comb = pd.concat([t1, t2], ignore_index=True)
    groups = E.make_group_ids(comb)
    rp = E.rp_label_from_xy(comb)
    scen = {"Medium_4Google": ["G1", "G2", "G3", "G4"],
            "High_4Google_3Linksys": ["G1", "G2", "G3", "G4", "L1", "L2", "L3"]}
    rows = []
    for sname, aps in scen.items():
        X, y = E.make_X_y(comb, aps)
        p = X.shape[1]
        for seed in SEEDS10:
            ss = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, rp))
            gs = list(GroupKFold(5).split(X, y, groups))
            preds = {}
            for ratio in (1.0, 0.7):
                mt = mtry_of(ratio, p)
                preds[("sample_cv", ratio)] = (y, oof(lambda: rf(mt, seed), X, y, ss), groups)
                preds[("rp_grouped_cv", ratio)] = (y, oof(lambda: rf(mt, seed), X, y, gs), groups)
                for a, b, lab in ((t1, t2, "d1_to_d2"), (t2, t1, "d2_to_d1")):
                    Xa, ya = E.make_X_y(a, aps)
                    Xb, yb = E.make_X_y(b, aps)
                    preds[(lab, ratio)] = (yb, rf(mt, seed).fit(Xa, ya).predict(Xb), E.make_group_ids(b))
            for proto in ("sample_cv", "rp_grouped_cv", "d1_to_d2", "d2_to_d1"):
                yt, p1, gg = preds[(proto, 1.0)]
                _, p7, _ = preds[(proto, 0.7)]
                rec = {"scenario": sname, "seed": seed, "protocol": proto,
                       "rmse_rho1": metrics(yt, p1)["rmse2d_m"], "rmse_rho07": metrics(yt, p7)["rmse2d_m"]}
                if seed == 0:
                    lo, hi = rp_bootstrap_rmse_diff(yt, p1, p7, gg)
                    rec.update({"diff_ci95_low_m": lo, "diff_ci95_high_m": hi})
                rows.append(rec)
            log(f"E11 {sname} seed={seed}")
            save(rows, "E11_eetac_multiseed.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="E1,E2,E3,E3b,E5,E6,E7,E8,E9,E11",
                    help="comma-separated parts; the default runs everything used in the paper")
    parts = ap.parse_args().parts.split(",")
    for part in ("E5", "E4", "E1", "E3", "E3b", "E2", "E6", "E7", "E8", "E9", "E11"):
        if part in parts:
            globals()[part.lower()]()
    log("done")


if __name__ == "__main__":
    main()
