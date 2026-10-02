"""
new_ideas.py
============

Screening experiments for candidate contributions beyond plain feature
subsampling.  Run from the project root:

    python scripts/screening/new_ideas.py --parts 12345 --datasets office,apartment,building

All point-accuracy results use the official RP-disjoint test split and,
where stated, 5-fold RP-grouped CV on TRAIN.  Results -> audit_results/N*.csv

1  Radio-map density x mtry   : does the best subspace ratio depend on RP density?
2  AP-paired subspace         : sample whole APs (RTT+RSS together) per tree
3  Geometry-aware features    : RTT multilateration + bias-cancelling differences
4  Residual RF                : multilateration + RF correction
5  Locally adaptive uncertainty: tree-spread ellipses vs global circle/ellipse
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts" / "original_dataset"))
import paper_all_contributions_wifi_rtt_rss as O  # noqa: E402

OUT = ROOT / "audit_results"
OUT.mkdir(exist_ok=True)
SEEDS = (0, 1, 2)

# AP coordinates in grid units, from dataset/Floor+office+apartment_AP_coords.txt
AP_COORDS = {
    "building": (
        [132, 126.2, 115, 88.2, 83, 72.8, 67.6, 53.4, 42.2, 31, 13, 3.3, -6],
        [1, 14.6, 6, 11.2, 2.3, 19, 17.5, 12, 1, 9.3, 14, -1.2, 12],
    ),
    "office": ([0, 5, 1.5], [-4, -2.3, 3.2]),
    "apartment": ([6.9, 7, 17.8, 13.4], [3, 11, 12.3, 6]),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rf(mf, seed=0, n=300):
    return RandomForestRegressor(n_estimators=n, max_features=mf, n_jobs=-1, random_state=seed)


def metrics(y, p, grid):
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
    # The coords file lists APs in the same order as the dataset's RTT columns
    # (office = AP1, AP6, AP7; apartment = AP2..AP5; building = AP1..AP13).
    ax, ay = AP_COORDS[name]
    assert len(ax) == len(rtt), f"{name}: {len(ax)} AP coords vs {len(rtt)} RTT columns"
    aps = np.column_stack([ax, ay]).astype(float)
    return Xtr, ytr, Xte, yte, O.make_rp_groups(tr), aps, O.GRID_SIZE_M[name]


def grouped_oof(fit_predict, X, y, groups):
    pred = np.zeros_like(y, dtype=float)
    for tr, va in GroupKFold(5).split(X, y, groups):
        pred[va] = fit_predict(X[tr], y[tr], X[va])
    return pred


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def multilaterate(X, aps, grid, iters=15):
    """Weighted Gauss-Newton multilateration from RTT (mm) in grid units.

    Weights favour strong RSS (closer, more often LOS).  Returns position,
    number of usable APs and RMS range residual.
    """
    M = aps.shape[0]
    rtt, rss = X[:, :M], X[:, M:]
    valid = rtt < O.RTT_MISSING - 1
    d = np.clip(rtt, 0.0, None) / 1000.0 / grid
    w = np.where(valid, 10 ** ((np.clip(rss, -100, -30) + 100) / 20.0), 0.0)
    wsum = w.sum(1, keepdims=True)
    wsum[wsum == 0] = 1.0
    p = (w @ aps) / wsum
    for _ in range(iters):
        diff = p[:, None, :] - aps[None, :, :]
        rng = np.linalg.norm(diff, axis=2) + 1e-6
        J = diff / rng[..., None]
        r = rng - d
        JW = J * w[..., None]
        H = np.einsum("nmi,nmj->nij", JW, J) + 1e-3 * np.eye(2)
        g = np.einsum("nmi,nm->ni", JW, r)
        p = p - np.linalg.solve(H, g[..., None])[..., 0]
    rng = np.linalg.norm(p[:, None, :] - aps[None, :, :], axis=2)
    res = np.sqrt((w * (rng - d) ** 2).sum(1) / wsum[:, 0])
    return p, valid.sum(1), res


def geo_features(X, aps, grid):
    M = aps.shape[0]
    p, nvalid, res = multilaterate(X, aps, grid)
    rtt, rss = X[:, :M], X[:, M:]
    valid = rtt < O.RTT_MISSING - 1
    rtt_rel = np.where(valid, rtt - np.where(valid, rtt, np.inf).min(1, keepdims=True), O.RTT_MISSING)
    heard = rss > O.RSS_MISSING + 1
    rss_rel = np.where(heard, rss - np.where(heard, rss, -np.inf).max(1, keepdims=True), O.RSS_MISSING)
    return p, np.column_stack([p, nvalid, res]), np.column_stack([rtt_rel, rss_rel])


# -----------------------------------------------------------------------------
# Part 1: density x mtry
# -----------------------------------------------------------------------------

def part1(datasets):
    rows = []
    for name in datasets:
        Xtr, ytr, Xte, yte, g, _, grid = load(name)
        p = Xtr.shape[1]
        ug = np.unique(g)
        for frac in (1.0, 0.75, 0.5, 0.25):
            for rep in (0, 1):
                rs = np.random.default_rng(100 + rep)
                keep = ug if frac == 1.0 else rs.choice(ug, int(round(frac * len(ug))), replace=False)
                m = np.isin(g, keep)
                for ratio in (0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
                    mtry = max(1, int(round(ratio * p)))
                    pred = rf(mtry, rep).fit(Xtr[m], ytr[m]).predict(Xte)
                    rows.append({"dataset": name, "rp_fraction": frac, "rep": rep, "ratio": ratio,
                                 "mtry": mtry, **metrics(yte, pred, grid)})
                log(f"1 {name} frac={frac} rep={rep} done")
                if frac == 1.0:
                    break
        pd.DataFrame(rows).to_csv(OUT / "N1_density_mtry.csv", index=False)


# -----------------------------------------------------------------------------
# Part 2: AP-paired vs feature-level per-tree subspace (and per-node RF)
# -----------------------------------------------------------------------------

def _fit_tree(X, y, cols, seed):
    rs = np.random.default_rng(seed)
    idx = rs.integers(0, len(X), len(X))
    t = DecisionTreeRegressor(random_state=seed).fit(X[idx][:, cols], y[idx])
    return t, cols


def subspace_ensemble(Xtr, ytr, Xte, M, mode, ratio, seed, n=300):
    rs = np.random.default_rng(seed)
    specs = []
    for t in range(n):
        if mode == "ap":
            k = max(1, int(round(ratio * M)))
            a = rs.choice(M, k, replace=False)
            cols = np.sort(np.concatenate([a, a + M]))
        else:
            k = max(1, int(round(ratio * 2 * M)))
            cols = np.sort(rs.choice(2 * M, k, replace=False))
        specs.append((cols, seed * 100000 + t))
    trees = Parallel(n_jobs=-1)(delayed(_fit_tree)(Xtr, ytr, c, s) for c, s in specs)
    return np.mean([t.predict(Xte[:, c]) for t, c in trees], axis=0)


def part2(datasets):
    rows = []
    for name in datasets:
        Xtr, ytr, Xte, yte, g, aps, grid = load(name)
        M = aps.shape[0]
        for seed in SEEDS:
            cands = {
                "RF-node-F0.7": lambda a, b, c: rf(0.7, seed).fit(a, b).predict(c),
                "tree-feature-0.7": lambda a, b, c: subspace_ensemble(a, b, c, M, "feat", 0.7, seed),
                "tree-AP-0.7": lambda a, b, c: subspace_ensemble(a, b, c, M, "ap", 0.7, seed),
                "tree-AP-0.5": lambda a, b, c: subspace_ensemble(a, b, c, M, "ap", 0.5, seed),
            }
            for cname, fp in cands.items():
                test = metrics(yte, fp(Xtr, ytr, Xte), grid)
                cv = metrics(ytr, grouped_oof(fp, Xtr, ytr, g), grid)
                rows.append({"dataset": name, "seed": seed, "model": cname,
                             **{f"test_{k}": v for k, v in test.items()},
                             **{f"cv_{k}": v for k, v in cv.items()}})
                log(f"2 {name} s{seed} {cname} test={test['paper_metric_m']:.4f} cv={cv['paper_metric_m']:.4f}")
            pd.DataFrame(rows).to_csv(OUT / "N2_ap_subspace.csv", index=False)


# -----------------------------------------------------------------------------
# Parts 3+4: geometry features and residual RF
# -----------------------------------------------------------------------------

def part34(datasets):
    rows = []
    for name in datasets:
        Xtr, ytr, Xte, yte, g, aps, grid = load(name)
        ml_te, geo_te, diff_te = geo_features(Xte, aps, grid)
        log(f"3 {name} multilateration alone: {metrics(yte, ml_te, grid)}")

        def make(variant, mf, seed):
            def fp(a, b, c):
                ml_a, geo_a, diff_a = geo_features(a, aps, grid)
                ml_c, geo_c, diff_c = geo_features(c, aps, grid)
                if variant == "base":
                    A, C = a, c
                elif variant == "geo":
                    A, C = np.hstack([a, geo_a]), np.hstack([c, geo_c])
                else:  # geo+diff and residual use the richest input
                    A, C = np.hstack([a, geo_a, diff_a]), np.hstack([c, geo_c, diff_c])
                if variant == "residual":
                    return ml_c + rf(mf, seed).fit(A, b - ml_a).predict(C)
                return rf(mf, seed).fit(A, b).predict(C)
            return fp

        for seed in SEEDS:
            for variant in ("base", "geo", "geo+diff", "residual"):
                for mf in (1.0, 0.7):
                    fp = make(variant, mf, seed)
                    test = metrics(yte, fp(Xtr, ytr, Xte), grid)
                    cv = metrics(ytr, grouped_oof(fp, Xtr, ytr, g), grid)
                    rows.append({"dataset": name, "seed": seed, "variant": variant, "max_features": mf,
                                 **{f"test_{k}": v for k, v in test.items()},
                                 **{f"cv_{k}": v for k, v in cv.items()}})
                    log(f"3/4 {name} s{seed} {variant} F{mf} test={test['paper_metric_m']:.4f} "
                        f"cv={cv['paper_metric_m']:.4f} cv2d={cv['rmse2d_m']:.4f}")
                pd.DataFrame(rows).to_csv(OUT / "N34_geometry.csv", index=False)


# -----------------------------------------------------------------------------
# Part 5: locally adaptive uncertainty
# -----------------------------------------------------------------------------

def tree_cov(model, X, grid):
    P = np.stack([t.predict(X) for t in model.estimators_]) * grid  # (T, n, 2)
    P = P - P.mean(0, keepdims=True)
    return np.einsum("tni,tnj->nij", P, P) / (P.shape[0] - 1)


def cq(scores, conf):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, math.ceil((n + 1) * conf) / n), method="higher"))


def part5(datasets):
    rows = []
    for name in datasets:
        Xtr, ytr, Xte, yte, g, _, grid = load(name)
        for seed in SEEDS:
            fit, cal = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(Xtr, ytr, g))
            m = rf(0.7, seed).fit(Xtr[fit], ytr[fit])
            r_cal = (ytr[cal] - m.predict(Xtr[cal])) * grid
            r_te = (yte - m.predict(Xte)) * grid
            S_cal, S_te = tree_cov(m, Xtr[cal], grid), tree_cov(m, Xte, grid)
            # Global covariance from calibration-independent source: OOF on fit set.
            oof = grouped_oof(lambda a, b, c: rf(0.7, seed).fit(a, b).predict(c), Xtr[fit], ytr[fit], g[fit])
            Sig = np.cov(((ytr[fit] - oof) * grid).T)
            # Fixed (untuned) floor: median per-sample tree variance on the fit OOF scale.
            beta2 = float(np.median(np.trace(S_cal, axis1=1, axis2=2)) / 2.0)

            def maha(r, S):
                return np.sqrt(np.einsum("ni,nij,nj->n", r, np.linalg.inv(S), r))

            I = np.eye(2)
            methods = {
                "global-circle": (np.linalg.norm(r_cal, axis=1), np.linalg.norm(r_te, axis=1),
                                  lambda q: np.full(len(r_te), math.pi * q ** 2)),
                "global-ellipse": (maha(r_cal, np.broadcast_to(Sig, (len(r_cal), 2, 2))),
                                   maha(r_te, np.broadcast_to(Sig, (len(r_te), 2, 2))),
                                   lambda q: np.full(len(r_te), math.pi * q ** 2 * math.sqrt(np.linalg.det(Sig)))),
            }
            sc = np.sqrt(np.trace(S_cal, axis1=1, axis2=2) / 2 + beta2)
            st = np.sqrt(np.trace(S_te, axis1=1, axis2=2) / 2 + beta2)
            methods["local-circle"] = (np.linalg.norm(r_cal, axis=1) / sc, np.linalg.norm(r_te, axis=1) / st,
                                       lambda q: math.pi * (q * st) ** 2)
            Lc, Lt = S_cal + beta2 * I, S_te + beta2 * I
            methods["local-ellipse"] = (maha(r_cal, Lc), maha(r_te, Lt),
                                        lambda q: math.pi * q ** 2 * np.sqrt(np.linalg.det(Lt)))
            for conf in (0.90, 0.95):
                for mname, (s_cal, s_te, area_fn) in methods.items():
                    q = cq(s_cal, conf)
                    a = area_fn(q)
                    rows.append({"dataset": name, "seed": seed, "confidence": conf, "method": mname,
                                 "coverage": float(np.mean(s_te <= q)),
                                 "mean_area_m2": float(np.mean(a)), "median_area_m2": float(np.median(a)),
                                 "cal_rps": len(np.unique(g[cal]))})
            log(f"5 {name} s{seed} done")
            pd.DataFrame(rows).to_csv(OUT / "N5_local_uncertainty.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="12345")
    ap.add_argument("--datasets", default="office,apartment,building")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global SEEDS, OUT
    SEEDS = tuple(int(x) for x in a.seeds.split(","))
    if a.tag:
        OUT = OUT / a.tag
        OUT.mkdir(exist_ok=True)
    ds = a.datasets.split(",")
    if "3" in a.parts or "4" in a.parts:
        part34(ds)
    if "5" in a.parts:
        part5(ds)
    if "1" in a.parts:
        part1(ds)
    if "2" in a.parts:
        part2(ds)
    log("done")


if __name__ == "__main__":
    main()
