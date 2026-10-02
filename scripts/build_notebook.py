"""
build_notebook.py
=================

Generates reproduce.ipynb (Jupyter notebook, nbformat 4) in the repository
root.  Run from the repository root:

    python scripts/build_notebook.py

The notebook is generated rather than edited by hand so that its content stays
consistent with the scripts and can be reviewed as plain Python.
"""

import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# =============================================================================
md(r"""
# Reproducing the paper

**Feature Subsampling in Hybrid Wi-Fi RTT–RSS Fingerprinting: Radio-Map Density, Ratio Selection, and Calibration With Clustered Scans**
*Nabil Abdelkader Nouri (CSAAIL, University of Djelfa, Algeria) and Salim Naouri (Department of Computer Science, Southwest Jiaotong University, Chengdu, China)*

This notebook walks through every experiment of the paper in the order in which the results are presented, explains what each experiment tests, shows the key numbers, and checks that they match the values reported in the paper.

It runs in two modes:

| Mode | Setting | Needs the datasets | Time |
|---|---|---|---|
| **Quick** (default) | `RUN_EXPERIMENTS = False` | No | seconds |
| **Full** | `RUN_EXPERIMENTS = True` | Yes (see `dataset/README.md`) | ~2 h on a laptop CPU |

In quick mode, the notebook reads the experiment outputs shipped with the repository (`paper_results/`, `audit_results/`, `eetac_external_validation_results/`) and regenerates the paper's tables and figures from them. In full mode, it first reruns the experiments from the raw data and then does the same.

**Run this notebook from the repository root** (the folder that contains this file).
""")

md(r"""
## 0. Configuration
""")

code(r"""
# ---- Switches -----------------------------------------------------------------
RUN_EXPERIMENTS = False   # True: rerun the main pipeline and the extension experiments (~2 h, needs the data)
RUN_SCREENING   = False   # True: also rerun the screening/audit experiments of the supplementary (~2 h)
COMPILE_PAPER   = False   # True: compile paper/main.tex with latexmk (needs a LaTeX installation)

# ---- Imports and helpers --------------------------------------------------------
import os, sys, subprocess, shutil, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

try:
    from IPython.display import display, Markdown
except ImportError:                     # allows running the cells as a plain script
    display = lambda x: print(x.to_string() if hasattr(x, "to_string") else x)
    Markdown = lambda s: s

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path.cwd()
assert (ROOT / "run_all_paper_experiments.py").exists(), \
    "Start the notebook from the repository root (the folder that contains run_all_paper_experiments.py)."

EXT = ROOT / "audit_results" / "extensions"
MAIN = ROOT / "paper_results" / "tables"


def run(cmd):
    # Run a command from the repository root and stream its output.
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    tail = (proc.stdout or "").strip().splitlines()[-15:]
    print("\n".join(tail))
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


CHECKS = []


def check(name, ok, detail=""):
    # Record a comparison between a value in the paper and the value in the results.
    CHECKS.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""))


print("Repository root:", ROOT)
""")

# =============================================================================
md(r"""
## 1. Environment and data check

The results in the paper were produced with the package versions in `requirements.txt` on a laptop CPU (Intel Core i5-13500HX, 20 threads, 16 GB RAM, no GPU). Other recent versions should give the same conclusions; exact numbers can differ slightly between scikit-learn versions.
""")

code(r"""
import sklearn, scipy, joblib
installed = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
             "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
             "matplotlib": matplotlib.__version__}
pinned = {}
for line in (ROOT / "requirements.txt").read_text().splitlines():
    if "==" in line and not line.startswith("#"):
        name, ver = line.split("==")
        pinned[name.strip()] = ver.strip()
display(pd.DataFrame([{"package": k, "installed": v, "used for the paper": pinned.get(k, "")}
                      for k, v in installed.items()]))

required = [
    "database_building_train.csv", "database_building_test.csv",
    "database_office_train.csv", "database_office_test.csv",
    "database_apartment_train.csv", "database_apartment_test.csv",
    "EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 1.csv",
    "EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 1.csv",
    "EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 2.csv",
    "EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 2.csv",
]
present = {f: (ROOT / "dataset" / f).exists() for f in required}
DATA_AVAILABLE = all(present.values())
display(pd.DataFrame({"file": list(present), "present": list(present.values())}))
print("All datasets present:", DATA_AVAILABLE)
if RUN_EXPERIMENTS and not DATA_AVAILABLE:
    raise FileNotFoundError("RUN_EXPERIMENTS=True needs the datasets in dataset/ (see dataset/README.md).")
""")

# =============================================================================
md(r"""
## 2. Main comparison: $\rho=0.7$ versus the full-feature forest $\rho=1$

**What is tested.** A Random Forest regressor (300 trees) maps a hybrid RTT–RSS fingerprint to a 2D position. At every split it considers $m_{\mathrm{try}}=\lfloor\rho p\rfloor$ of the $p$ features. Prior work uses $\rho=1$ (every feature at every split, i.e., bagged trees). The proposed configuration uses the fixed ratio $\rho=0.7$; everything else is identical.

**How.** `run_all_paper_experiments.py` runs two engines:

* `scripts/original_dataset/` — the Building / Office / Apartment benchmark: official reference-point (RP)-disjoint test split, 5-fold RP-grouped cross-validation (pooled out-of-fold predictions), paired cluster bootstrap over test RPs, uncertainty regions, and timing;
* `scripts/eetac/` — the independent EETAC dataset: sample-level CV, RP-grouped CV, and cross-day transfer, after aligning the two acquisition days (Section 3).

It writes the main-comparison results (paper Tables 4, 6, 7 and supplementary Tables S7–S10, Figs. S1–S7) to `paper_results/`. Seed: 42.

Two error metrics are reported. The *paper-compatible* metric is the mean of the per-coordinate RMSEs times the grid spacing, as used by the published baseline; the *2D RMSE* is the usual Euclidean RMSE. For isotropic errors, the 2D RMSE is about $\sqrt{2}$ times larger.
""")

code(r"""
if RUN_EXPERIMENTS:
    run([sys.executable, "run_all_paper_experiments.py"])

t3 = pd.read_csv(MAIN / "table03_original_accuracy.csv")
display(t3.round(3))

impr = dict(zip(t3["Dataset"], t3["Improvement vs F1.0 (%)"]))
check("Error reduction 14.3 / 7.1 / 5.1 % (Building / Office / Apartment)",
      [round(impr[d], 1) for d in ("Building", "Office", "Apartment")] == [14.3, 7.1, 5.1],
      ", ".join(f"{d} {impr[d]:.2f}%" for d in impr))
ci = t3.set_index("Dataset")[["Diff 95% CI low (m)", "Diff 95% CI high (m)"]]
check("Bootstrap CI excludes zero for Building and Office, includes zero for Apartment",
      ci.loc["Building"].max() < 0 and ci.loc["Office"].max() < 0
      and ci.loc["Apartment", "Diff 95% CI low (m)"] < 0 < ci.loc["Apartment", "Diff 95% CI high (m)"])
check("True 2D RMSE of the reproduced Building baseline is 0.851 m (published metric: 0.60 m)",
      round(t3.set_index("Dataset").loc["Building", "2D RMSE F1.0 (m)"], 3) == 0.851)
""")

code(r"""
t4 = pd.read_csv(MAIN / "table04_grouped_cv.csv")
print("RP-grouped cross-validation (pooled out-of-fold predictions):")
display(t4.round(3))
""")

# =============================================================================
md(r"""
## 3. EETAC: acquisition-day frame alignment

**Why this matters.** The EETAC dataset was recorded on two days. Comparing, for every RP, the median RTT fingerprint of the four Google APs on Day 1 with all Day 2 fingerprints shows that the two days use **vertically mirrored coordinate frames**: the Day 2 position $y$ corresponds to $8-y$ on Day 1. Without correcting this, every experiment that pools the days or trains on one day and tests on the other assigns one coordinate label to two different physical locations.

The EETAC engine detects the mapping from the data (`detect_day_frame`), applies $y\mapsto 8-y$ to Day 2, and writes the evidence below. The central row $y=4$ m is identical under both mappings and therefore carries no evidence.
""")

code(r"""
audit = pd.read_csv(ROOT / "eetac_external_validation_results" / "day_frame_alignment_audit.csv")
display(audit)
check("Mirrored frame detected: 22 of 25 RPs match y -> 8 - y",
      int(audit.loc[0, "mirror_y_matches"]) == 22 and audit.loc[0, "selected"] == "mirror_y")

if DATA_AVAILABLE:
    # Live recomputation from the raw files, plus one concrete example.
    sys.path.insert(0, str(ROOT / "scripts" / "eetac"))
    import warnings; warnings.filterwarnings("ignore")
    import eetac_external_validation_subspace_rf_v2 as eetac
    d1, _ = eetac.build_day_table(1)
    d2, _ = eetac.build_day_table(2)
    display(eetac.detect_day_frame(d1, d2))
    cols = ["RTT_G1", "RTT_G2", "RTT_G3", "RTT_G4"]
    example = pd.DataFrame({
        "Day 1 at (0, 0)": d1[(d1.x == 0) & (d1.y == 0)][cols].median(),
        "Day 2 at (0, 0)": d2[(d2.x == 0) & (d2.y == 0)][cols].median(),
        "Day 2 at (0, 8)": d2[(d2.x == 0) & (d2.y == 8)][cols].median(),
    }).round(1)
    print("Median RTT (m) of the four Google APs: Day 1 (0,0) matches Day 2 (0,8), not Day 2 (0,0)")
    display(example)
""")

code(r"""
flip = pd.read_csv(ROOT / "audit_results" / "C_eetac_flip.csv")
cols = ["sample_cv_rmse2d", "rp_grouped_cv_rmse2d", "d1_to_d2_rmse2d", "d2_to_d1_rmse2d"]
summary = flip.groupby(["y_flip_day2", "scenario", "max_features"])[cols].mean().round(3)
summary.index = summary.index.set_names(["frames aligned", "scenario", "rho"])
print("EETAC 2D RMSE (m), mean of seeds 0-2, without and with frame alignment:")
display(summary)

# EETAC results of the main pipeline (seed 42, aligned)
t5 = pd.read_csv(MAIN / "table05_eetac_external_validation.csv")
t6 = pd.read_csv(MAIN / "table06_cross_day.csv")
display(t5.round(3)); display(t6.round(3))

check("Aligned cross-day errors of both models are 1.2-1.7 m (seed 42)",
      t6[["F0.7 2D RMSE (m)", "F1.0 2D RMSE (m)"]].stack().between(1.2, 1.7).all())
check("Without alignment, cross-day errors exceed 5 m",
      (flip[~flip.y_flip_day2][["d1_to_d2_rmse2d", "d2_to_d1_rmse2d"]] > 5).all().all())
""")

# =============================================================================
md(r"""
## 4. Extension experiments E1–E7

`scripts/extensions/paper_extensions.py` runs seven analyses with the same data and Random Forest implementation as the main pipeline, including $m_{\mathrm{try}}=\lfloor\rho p\rfloor$. They use seeds 0–9 or 0–2 (the main pipeline uses 42), so the same configuration can differ slightly between the main and the extension tables.

| Part | Question | Paper |
|---|---|---|
| E1 | How does the best $\rho$ depend on radio-map density (Random Forest)? | density figure, RF/ET density table |
| E2 | Which rule should choose $\rho$: OOB, sample-level CV, RP-grouped CV, or a fixed default? | ratio-selection table, Fig. S8 |
| E3, E3b | How do Breiman's $p/3$, extremely randomized trees, Rotation Forest, gradient boosting, and WKNN compare? | literature table |
| E4 | Single-split-style calibration statistics (superseded by E8) | — |
| E5 | Does the EETAC dataset show the same ratio behaviour? | Table S3 |
| E6 | How small can the forest be made? | model-size figure, Table S11 |
| E7 | Is the density effect specific to Random Forests (extremely randomized trees)? | RF/ET density table |
| E8 | Scan-level vs. group (one scan per RP) vs. RP-count conformal calibration | calibration table |
| E9, E11 | Is $\rho=0.7$ vs. $\rho=1$ robust over ten seeds (official tests, EETAC with bootstrap CIs)? | robustness table |
""")

code(r"""
if RUN_EXPERIMENTS:
    run([sys.executable, "scripts/extensions/paper_extensions.py",
         "--parts", "E1,E2,E3,E3b,E5,E6,E7,E8,E9,E11"])   # several hours
print("Extension results:", sorted(p.name for p in EXT.glob("*.csv")))
""")

md(r"""
### 4.1 Radio-map density (E1, E7)

The training radio map is thinned to 100 %, 75 %, 50 %, and 25 % of its RPs (complete RPs are removed; ten random draws), and every ratio $\rho\in\{0.2,0.35,0.5,0.7,0.85,1.0\}$ is evaluated on the full official test set. The same thinned maps are used for Random Forests (E1) and extremely randomized trees (E7).

**Finding.** For Random Forests, $\rho=1$ is never the best ratio, and the best ratio decreases as the map becomes sparser in all three environments; the gain over $\rho=1$ reaches 25.5 %. Extremely randomized trees, which already draw random split thresholds, gain at most 0.8 % from feature subsampling.
""")

code(r"""
e1 = pd.read_csv(EXT / "E1_density_ratio.csv")
e7 = pd.read_csv(EXT / "E7_density_ratio_extratrees.csv")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, name in zip(axes, ("building", "apartment", "office")):
    for frac, g in e1[e1.dataset == name].groupby("rp_fraction"):
        s = g.groupby("ratio").paper_metric_m.mean()
        ax.plot(s.index, s.values, marker="o", label=f"{int(frac*100)}% RPs")
        ax.plot(s.idxmin(), s.min(), "k*", ms=12)
    ax.set_title(f"Random Forest - {name}"); ax.set_xlabel("rho"); ax.grid(alpha=0.3)
axes[0].set_ylabel("paper-compatible error (m)"); axes[-1].legend(fontsize=8)
plt.tight_layout(); plt.show()

rows = []
for model, d in (("Random Forest", e1), ("Extremely randomized trees", e7)):
    for (name, frac), g in d.groupby(["dataset", "rp_fraction"]):
        s = g.groupby("ratio").paper_metric_m.mean()
        rows.append({"model": model, "dataset": name, "RP fraction": frac,
                     "best rho": s.idxmin(), "best error (m)": round(s.min(), 3),
                     "gain over rho=1 (%)": round(100 * (s.loc[1.0] - s.min()) / s.loc[1.0], 1)})
dens = pd.DataFrame(rows)
display(dens.pivot_table(index=["dataset", "RP fraction"], columns="model",
                         values=["best rho", "gain over rho=1 (%)"]))

rf = dens[dens.model == "Random Forest"].set_index(["dataset", "RP fraction"])
et = dens[dens.model == "Extremely randomized trees"]
check("RF: rho=1 is never the best ratio", (rf["best rho"] < 1.0).all())
check("RF: best rho is smaller at 25% than at full density in all three environments",
      all(rf.loc[(n, 0.25), "best rho"] < rf.loc[(n, 1.0), "best rho"] for n in ("building", "apartment", "office")))
check("RF Building: best rho 0.5 at full density, 0.2 at 25%",
      rf.loc[("building", 1.0), "best rho"] == 0.5 and rf.loc[("building", 0.25), "best rho"] == 0.2)
check("RF: gain over rho=1 reaches 25.5% (Building, 50%) and 18.0% (Apartment, 25%)",
      rf.loc[("building", 0.5), "gain over rho=1 (%)"] == 25.5
      and rf.loc[("apartment", 0.25), "gain over rho=1 (%)"] == 18.0)
check("Extremely randomized trees gain at most 0.8% from subsampling",
      et["gain over rho=1 (%)"].max() <= 0.8, f"max {et['gain over rho=1 (%)'].max()}%")
""")

md(r"""
### 4.2 Choosing the ratio (E2)

For each ratio, three validation estimates are computed on the training file — the out-of-bag (OOB) error, 5-fold sample-level CV, and 5-fold RP-grouped CV — together with the official test error. Each selection rule picks the ratio with the lowest validation error; its **regret** is how much worse that ratio is on the test set than the best ratio. The test set only scores the rules; it never chooses $\rho$.

**Finding.** OOB and sample-level CV are an order of magnitude too optimistic because every scan is evaluated by trees that saw other scans of the same RP; RP-grouped CV trains on a sparser map and, because of the density effect, prefers ratios that are too small. The fixed default $\rho=0.7$ stays within 4.1 % of the best ratio everywhere.
""")

code(r"""
e2 = pd.read_csv(EXT / "E2_selection_curves.csv")
best = e2.groupby("dataset").test_paper_metric_m.min()

def regret(name, ratio):
    g = e2[e2.dataset == name]
    return 100 * (g[np.isclose(g.ratio, ratio)].test_paper_metric_m.iloc[0] / best[name] - 1)

rules = {}
for label, col in (("OOB", "oob_paper_metric_m"), ("sample-level CV", "sample_cv_paper_metric_m"),
                   ("RP-grouped CV", "group_cv_paper_metric_m")):
    rules[label] = {n: e2.loc[e2[e2.dataset == n][col].idxmin(), "ratio"] for n in best.index}
for r in (1.0, 0.7, 0.5, 0.35):
    rules[f"fixed rho={r}"] = {n: r for n in best.index}
tab = pd.DataFrame({rule: {n: round(regret(n, picks[n]), 1) for n in picks} for rule, picks in rules.items()}).T
tab["worst (%)"] = tab.max(axis=1)
print("Test regret (%) of each selection rule:")
display(tab)

check("Fixed rho=0.7: worst-case regret 4.1%", round(tab.loc["fixed rho=0.7", "worst (%)"], 1) == 4.1)
check("Fixed rho=1 (bagging): worst-case regret 19.7%", round(tab.loc["fixed rho=1.0", "worst (%)"], 1) == 19.7)
check("RP-grouped CV: worst-case regret 22.5%", round(tab.loc["RP-grouped CV", "worst (%)"], 1) == 22.5)
""")

md(r"""
### 4.3 Literature baselines (E3, E3b)

All models are trained on the full training file and evaluated on the official test set and with RP-grouped CV (three seeds for Office and Apartment, one for Building).
""")

code(r"""
lit = pd.concat([pd.read_csv(EXT / "E3_literature.csv"), pd.read_csv(EXT / "E3b_gradient_boosting.csv")])
lt = lit.groupby(["model", "dataset"])[["test_paper_metric_m", "cv_paper_metric_m"]].mean().unstack().round(3)
display(lt)
t = lit.groupby(["model", "dataset"]).test_paper_metric_m.mean().unstack()
check("rho=0.7 is best on the Office and Apartment official tests",
      t["office"].idxmin() == "RF-F0.7" and t["apartment"].idxmin() == "RF-F0.7")
check("Gradient boosting is less accurate than rho=0.7 on every official test",
      (t.loc["HistGradientBoosting"] > t.loc["RF-F0.7"]).all())
""")

md(r"""
### 4.4 Compact forests (E6)

The accuracy gained by subsampling can be spent on a smaller model. Forests with 50–300 trees, larger leaves, and depth limits are compared by serialized size and single-thread inference time.
""")

code(r"""
e6 = pd.read_csv(EXT / "E6_model_size.csv")
e6["max_depth"] = e6.max_depth.fillna("None").astype(str)
sel = e6[(e6.min_samples_leaf == 1) & (e6.max_depth == "None")
         & (((e6.ratio == 1.0) & (e6.n_trees == 300)) | ((e6.ratio == 0.7) & e6.n_trees.isin([50, 100, 300])))]
display(sel[["dataset", "ratio", "n_trees", "size_mb", "infer_us_per_scan_1thread", "train_s", "paper_metric_m"]].round(3))

ratios = []
for name, g in sel.groupby("dataset"):
    base = g[(g.ratio == 1.0)].iloc[0]
    for _, r in g[(g.ratio == 0.7) & g.n_trees.isin([50, 100])].iterrows():
        assert r.paper_metric_m < base.paper_metric_m
        ratios.append(base.size_mb / r.size_mb)
check("50-100-tree rho=0.7 forests are more accurate than the 300-tree rho=1 forest and 2.5-5.5x smaller",
      round(min(ratios), 1) == 2.5 and round(max(ratios), 1) == 5.5, f"{min(ratios):.2f}-{max(ratios):.2f}x")
""")

md(r"""
### 4.5 Calibration with clustered scans (E8)

Split-conformal prediction calibrates a region on held-out data. In fingerprinting, the held-out data are a few RPs with about 120 near-identical scans each, so the exchangeable units are the $G$ RPs, not the scans. E8 repeats the calibration over 20 random RP-level splits (10 for Building) and compares three rules on the same splits and models: scan-level (standard), group (one random scan per RP, valid when RPs are exchangeable; needs $G\ge 1/\alpha-1$), and RP-count (pooled scores, level computed from $G$; a conservative heuristic). The guarantee concerns the **mean** coverage over calibration draws.
""")

code(r"""
e8 = pd.read_csv(EXT / "E8_group_conformal.csv")
e8["under"] = e8.coverage < e8.confidence
cal = e8.groupby(["dataset", "cal_rps", "rule", "confidence"]).agg(
    mean_coverage=("coverage", "mean"), min_coverage=("coverage", "min"),
    under_pct=("under", lambda s: 100 * s.mean()), mean_area_m2=("area_m2", "mean")).round(3)
display(cal)

m = cal.mean_coverage
check("Apartment (G=21): scan-level mean coverage below nominal at 90% and 95%",
      m[("apartment", 21, "scan-level", 0.9)] < 0.9 and m[("apartment", 21, "scan-level", 0.95)] < 0.95)
check("Apartment (G=21): group calibration attains nominal mean coverage (0.916 / 0.947)",
      round(m[("apartment", 21, "group-subsample", 0.9)], 3) == 0.916
      and round(m[("apartment", 21, "group-subsample", 0.95)], 3) == 0.947)
a = cal.mean_area_m2
inc = [a[("apartment", 21, "group-subsample", c)] / a[("apartment", 21, "scan-level", c)] - 1 for c in (0.9, 0.95)]
check("Apartment: group regions are 22-30% larger than scan-level regions",
      round(100 * min(inc)) == 22 and round(100 * max(inc)) == 30, f"{inc}")
check("Office (G=7): group calibration gives no finite region",
      np.isinf(e8[(e8.dataset == "office") & (e8.rule == "group-subsample")].area_m2).all())
check("RP-count at 95% in Apartment over-covers (0.995, 33.2 m2)",
      round(m[("apartment", 21, "RP-count", 0.95)], 3) == 0.995 and round(a[("apartment", 21, "RP-count", 0.95)], 1) == 33.2)

""")

md(r"""
### 4.6 Robustness over ten seeds (E9, E11)

The comparison $\rho=0.7$ vs. $\rho=1$ is repeated with seeds 0–9 on the official tests (E9) and on all eight EETAC protocols (E11, with paired RP-clustered bootstrap 95 % intervals of the 2D RMSE difference for seed 0).
""")

code(r"""
e9 = pd.read_csv(EXT / "E9_multiseed_official.csv")
w = e9.pivot_table(index=["dataset", "seed"], columns="ratio", values=["paper_metric_m", "p90_m"])
imp = 100 * (1 - w[("paper_metric_m", 0.7)] / w[("paper_metric_m", 1.0)])
display(imp.groupby("dataset").agg(["mean", "std"]).round(1))
check("rho=0.7 has the lower error in all 30 seed-environment pairs",
      (w[("paper_metric_m", 0.7)] < w[("paper_metric_m", 1.0)]).all())
check("Mean improvement 14.4 / 9.2 / 5.2 % (Building / Office / Apartment)",
      [round(imp.xs(n).mean(), 1) for n in ("building", "office", "apartment")] == [14.4, 9.2, 5.2])
p90_worse = (w[("p90_m", 0.7)] > w[("p90_m", 1.0)]).xs("apartment")
check("Apartment P90 is worse with rho=0.7 in all ten seeds", p90_worse.all())

e11 = pd.read_csv(EXT / "E11_eetac_multiseed.csv")
e11["win"] = e11.rmse_rho07 < e11.rmse_rho1
display(e11.groupby(["scenario", "protocol"]).agg(wins=("win", "sum"),
        ci_low=("diff_ci95_low_m", "first"), ci_high=("diff_ci95_high_m", "first")).round(3))
check("rho=0.7 beats rho=1 in 74 of 80 EETAC seed-protocol comparisons",
      e11.win.sum() == 74 and len(e11) == 80)
tie = e11[(e11.scenario.str.startswith("High")) & (e11.protocol == "d1_to_d2")]
check("High Day1->Day2 is a tie: 4/10 wins, bootstrap CI includes zero",
      tie.win.sum() == 4 and (tie.diff_ci95_low_m.iloc[0] < 0 < tie.diff_ci95_high_m.iloc[0]))
""")

# =============================================================================
md(r"""
## 5. Screening experiments (supplementary material, optional)

`scripts/screening/` contains the audit and screening runs whose results appear in the supplementary material: the ratio sweep with three seeds, EETAC with and without frame alignment, per-tree and AP-paired subspaces, geometry features from RTT multilateration, a residual model, and locally adaptive conformal regions. They are not needed for the main paper.
""")

code(r"""
if RUN_SCREENING:
    run([sys.executable, "scripts/screening/claims_audit.py"])
    run([sys.executable, "scripts/screening/new_ideas.py", "--tag", "small", "--datasets", "office,apartment"])
    run([sys.executable, "scripts/screening/new_ideas.py", "--tag", "building", "--datasets", "building", "--seeds", "0"])
else:
    print("Screening experiments skipped (RUN_SCREENING = False); using the saved results in audit_results/.")
""")

# =============================================================================
md(r"""
## 6. Regenerating the paper's tables and figures

These scripts write LaTeX tables and PDF/PNG figures directly into `paper/`. They only read the result files, so they run in seconds in both modes.
""")

code(r"""
run([sys.executable, "scripts/extensions/make_paper_tables.py"])
run([sys.executable, "scripts/extensions/make_extension_figures.py"])
run([sys.executable, "scripts/extensions/make_supplementary.py"])

# Figures of the main pipeline (supplementary Figs. S1-S7)
dst = ROOT / "paper" / "figures" / "results"
for f in sorted((ROOT / "paper_results" / "figures").glob("fig*")):
    shutil.copy2(f, dst / f.name)
print("Tables:", sorted(p.name for p in (ROOT / "paper" / "tables").glob("*.tex")))
print("Figures:", sorted(p.name for p in dst.glob("*.pdf")))
""")

code(r"""
if COMPILE_PAPER:
    if shutil.which("latexmk") is None:
        print("latexmk not found; install a LaTeX distribution (e.g., TeX Live or MiKTeX).")
    else:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=ROOT / "paper")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "supplementary_material.tex"], cwd=ROOT / "paper" / "supplementary")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "supplementary_material.tex"], cwd=ROOT / "paper" / "supplementary")
        print("Compiled paper/main.pdf and paper/supplementary/supplementary_material.pdf")
else:
    print("To compile: cd paper && latexmk -pdf main.tex")
""")

# =============================================================================
md(r"""
## 7. Summary of the consistency checks

Each check compares a number stated in the paper with the value computed from the result files. In full mode, a failed check means that the rerun produced different numbers from the published ones (for example because of a different scikit-learn version), and the corresponding statements in the paper should be reviewed.
""")

code(r"""
summary = pd.DataFrame(CHECKS)
display(summary)
print(f"{(summary.status == 'PASS').sum()} of {len(summary)} checks passed.")
""")

md(r"""
## Notes

* **Run time** (laptop CPU, 20 threads): main pipeline ≈ 10 min; extension experiments ≈ 1.5 h; screening ≈ 2 h.
* **Seeds**: 42 for the main pipeline, 0–2 for the extensions; differences between the two are seed variability (≤ 0.013 m on Building).
* **Ratio definition**: $m_{\mathrm{try}}=\max(1,\lfloor\rho p\rfloor)$ everywhere, which is also what scikit-learn uses for a float `max_features`.
* **Large file**: `grouped_oof_residuals.csv` (9.5 MB) is regenerated by the main pipeline and is not stored in the repository.
""")


# =============================================================================
nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
out = Path("reproduce.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("written", out.resolve(), "with", len(CELLS), "cells")
