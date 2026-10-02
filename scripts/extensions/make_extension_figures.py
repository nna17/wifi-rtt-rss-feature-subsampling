"""
make_extension_figures.py
=========================

Builds the manuscript figures for the extension experiments from the CSVs in
audit_results/extensions/.  Run from the project root after
paper_extensions.py:

    python scripts/extensions/make_extension_figures.py

Writes PDF + PNG into paper/figures/results/:
    fig13_density_ratio      best subspace ratio vs radio-map density
    fig14_ratio_selection    error vs ratio for test / grouped CV / OOB
    fig15_model_size         accuracy vs serialized size of compact forests
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / "audit_results" / "extensions"
DST = ROOT / "paper" / "figures" / "results"
DST.mkdir(parents=True, exist_ok=True)

ORDER = ("building", "apartment", "office")
NICE = {"building": "Building", "apartment": "Apartment", "office": "Office"}


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(DST / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(DST / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig13():
    d = pd.read_csv(SRC / "E1_density_ratio.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name in zip(axes, ORDER):
        g = d[d.dataset == name]
        for frac, gg in g.groupby("rp_fraction"):
            s = gg.groupby("ratio").paper_metric_m.agg(["mean", "std"])
            rps = int(gg.train_rps.iloc[0])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", ms=4, capsize=2,
                        label=f"{int(frac * 100)}% RPs ({rps})")
            best = s["mean"].idxmin()
            ax.plot(best, s.loc[best, "mean"], marker="*", ms=13, color="k", zorder=5)
        ax.set_title(NICE[name])
        ax.set_xlabel(r"Subspace ratio $\rho = m_{\mathrm{try}}/p$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Paper-compatible error (m), official test")
    axes[-1].legend(fontsize=8, title="Training radio map", title_fontsize=8)
    save(fig, "fig13_density_ratio")


def fig14():
    d = pd.read_csv(SRC / "E2_selection_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [("test_paper_metric_m", "Official RP-disjoint test", "o"),
              ("group_cv_paper_metric_m", "5-fold RP-grouped CV (train)", "s"),
              ("oob_paper_metric_m", "Out-of-bag (sample level)", "^")]
    for ax, name in zip(axes, ORDER):
        g = d[d.dataset == name].sort_values("ratio")
        for col, label, mk in series:
            ax.plot(g.ratio, g[col], marker=mk, label=label)
        ax.axvline(0.7, ls="--", lw=1, color="gray")
        ax.set_title(NICE[name])
        ax.set_xlabel(r"Subspace ratio $\rho$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Paper-compatible error (m)")
    axes[0].legend(fontsize=8)
    save(fig, "fig14_ratio_selection")


def fig15():
    d = pd.read_csv(SRC / "E6_model_size.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name in zip(axes, ORDER):
        g = d[d.dataset == name]
        for ratio, gg in g.groupby("ratio"):
            gg = gg.sort_values("size_mb")
            front, best = [], np.inf
            for _, r in gg.iterrows():
                if r.paper_metric_m < best:
                    best = r.paper_metric_m
                    front.append(r)
            front = pd.DataFrame(front)
            ax.scatter(gg.size_mb, gg.paper_metric_m, s=10, alpha=0.3)
            ax.plot(front.size_mb, front.paper_metric_m, marker="o", ms=4,
                    label=rf"$\rho={ratio}$ (Pareto front)")
        ax.set_xscale("log")
        ax.set_title(NICE[name])
        ax.set_xlabel("Serialized model size (MB, log scale)")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Paper-compatible error (m), official test")
    axes[0].legend(fontsize=8)
    save(fig, "fig15_model_size")


# -----------------------------------------------------------------------------
# Tables (LaTeX, booktabs, written to paper/tables/)
# -----------------------------------------------------------------------------

TAB = ROOT / "paper" / "tables"


def write_table(stem, body):
    (TAB / f"{stem}.tex").write_text(body, encoding="utf-8")


def table10():
    """Test regret of ratio-selection rules (E2)."""
    d = pd.read_csv(SRC / "E2_selection_curves.csv")
    best = d.groupby("dataset").test_paper_metric_m.min()

    def regret(name, ratio):
        g = d[d.dataset == name]
        v = g[np.isclose(g.ratio, ratio)].test_paper_metric_m.iloc[0]
        return 100 * (v / best[name] - 1)

    rules = []
    for label, col in (("Out-of-bag error (sample level)", "oob_paper_metric_m"),
                       ("5-fold sample-level CV", "sample_cv_paper_metric_m"),
                       ("5-fold RP-grouped CV", "group_cv_paper_metric_m")):
        picks = {n: float(d.loc[d[d.dataset == n][col].idxmin(), "ratio"]) for n in ORDER}
        rules.append((label, picks))
    loeo = {}
    for n in ORDER:
        o = d[d.dataset != n].copy()
        o["reg"] = o.groupby("dataset").group_cv_paper_metric_m.transform(lambda s: s / s.min() - 1)
        loeo[n] = float(o.groupby("ratio").reg.max().idxmin())
    rules.append(("Leave-one-environment-out minimax (RP-grouped CV)", loeo))
    for r in (1.0, 0.7, 0.5, 0.35):
        lab = {1.0: r"Fixed $\rho=1.0$ (bagging; Feng et al.)",
               0.7: r"Fixed $\rho=0.7$ (this work)",
               0.5: r"Fixed $\rho=0.5$",
               0.35: r"Fixed $\rho=0.35\approx1/3$ (Breiman)"}[r]
        rules.append((lab, {n: r for n in ORDER}))

    lines = []
    for label, picks in rules:
        regs = [regret(n, picks[n]) for n in ORDER]
        cells = " & ".join(f"{picks[n]:.2f} & {regret(n, picks[n]):.1f}" for n in ORDER)
        lines.append(f"{label} & {cells} & {max(regs):.1f} \\\\")
    body = r"""\begin{table*}[!t]
\centering
\caption{Selecting the subspace ratio. For each rule, the selected $\rho$ and the resulting regret on the official RP-disjoint test set (percentage above the best ratio of the sweep, $\rho\in\{0.2,0.35,0.5,0.7,0.85,1.0\}$, paper-compatible metric, seed 0). The test set is used only to score the rules, never to select $\rho$.}
\label{tab:ratio_selection}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{2}{c}{Building} & \multicolumn{2}{c}{Apartment} & \multicolumn{2}{c}{Office} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Selection rule & $\rho$ & Regret (\%) & $\rho$ & Regret (\%) & $\rho$ & Regret (\%) & Worst (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table10_ratio_selection", body)


def table11():
    """Literature baselines (E3): mean over seeds."""
    d = pd.read_csv(SRC / "E3_literature.csv")
    if (SRC / "E3b_gradient_boosting.csv").exists():
        d = pd.concat([d, pd.read_csv(SRC / "E3b_gradient_boosting.csv")], ignore_index=True)
    order = ["RF-F1.0 (bagging)", "RF-F0.7", "RF-p/3 (Breiman)", "ExtraTrees-F1.0",
             "RotationForest", "HistGradientBoosting", "WKNN-k5"]
    order = [o for o in order if o in set(d.model)]
    label = {"RF-F1.0 (bagging)": r"RF, $\rho=1.0$ (bagging) \cite{feng2026robust}",
             "RF-F0.7": r"RF, $\rho=0.7$ (this work)",
             "RF-p/3 (Breiman)": r"RF, $m_{\mathrm{try}}=\lfloor p/3\rfloor$ \cite{breiman2001random}",
             "ExtraTrees-F1.0": r"Extremely randomized trees \cite{geurts2006extremely}",
             "RotationForest": r"Rotation Forest \cite{rodriguez2006rotation}",
             "HistGradientBoosting": r"Gradient-boosted trees (500 iterations)",
             "WKNN-k5": r"WKNN, $k=5$ \cite{bahl2000radar}"}
    m = d.groupby(["dataset", "model"])[["test_paper_metric_m", "cv_paper_metric_m"]].mean()
    best_t = m.test_paper_metric_m.groupby(level=0).min()
    best_c = m.cv_paper_metric_m.groupby(level=0).min()

    def cell(n, mod, col, best):
        v = m.loc[(n, mod), col]
        s = f"{v:.3f}"
        return rf"\textbf{{{s}}}" if np.isclose(v, best[n]) else s

    rows = []
    for mod in order:
        cells = " & ".join(f"{cell(n, mod, 'test_paper_metric_m', best_t)} & "
                           f"{cell(n, mod, 'cv_paper_metric_m', best_c)}" for n in ORDER)
        rows.append(f"{label[mod]} & {cells} \\\\")
    body = r"""\begin{table*}[!t]
\centering
\caption{Comparison with tree-ensemble and fingerprinting alternatives from the literature (paper-compatible metric, m). Test: official RP-disjoint test set. CV: pooled 5-fold RP-grouped CV on the training RPs. Office and Apartment: mean of three seeds; Building: one seed. Best value per column in bold.}
\label{tab:literature_baselines}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Building} & \multicolumn{2}{c}{Apartment} & \multicolumn{2}{c}{Office} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & Test & CV & Test & CV & Test & CV \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table11_literature_baselines", body)


def table12():
    """Calibration rules (E8), circle score, over random RP splits:
    scan-level, valid group (one scan per RP), and RP-count heuristic."""
    d = pd.read_csv(SRC / "E8_group_conformal.csv")
    label = {"scan-level": "Scan-level (standard)",
             "group-subsample": "Group, one scan per RP (valid)",
             "RP-count": "RP-count (conservative heuristic)"}
    rows = []
    for n in ORDER:
        g = d[d.dataset == n]
        G = int(g.cal_rps.iloc[0])
        nsp = g.split.nunique()
        first = True
        for rule in ("scan-level", "group-subsample", "RP-count"):
            cells = []
            for conf in (0.90, 0.95):
                h = g[(g.rule == rule) & np.isclose(g.confidence, conf)]
                fin = np.isfinite(h.area_m2)
                if fin.sum() == 0:
                    cells.append(r"\multicolumn{3}{c}{no finite region ($G<1/\alpha-1$)}")
                else:
                    cells.append(f"{h.coverage.mean():.3f} & {h.coverage.min():.3f} & {h.area_m2[fin].mean():.2f}")
            lead = f"{NICE[n]} ($G={G}$, {nsp} splits)" if first else ""
            rows.append(f"{lead} & {label[rule]} & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\midrule")
    body = r"""\begin{table*}[!t]
\centering
\caption{Split-conformal calibration of circular regions when the $\sim$120 scans of each RP are correlated, over repeated random RP-level fit/calibration splits ($25\%$ of training RPs for calibration; coverage on the official test set). Scan-level: all calibration scans treated as exchangeable. Group: one randomly chosen scan per calibration RP, with ordinary split conformal on these $G$ scores, which is valid when RPs are exchangeable~\cite{Dunn02102023}. RP-count: pooled scores with the finite-sample level computed from $G$. The guarantee of a valid method concerns the mean coverage over calibration draws; individual splits vary around it.}
\label{tab:calibration_rules}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{$1-\alpha=0.90$} & \multicolumn{3}{c}{$1-\alpha=0.95$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Dataset & Calibration rule & Mean cov. & Min cov. & Area (m$^2$) & Mean cov. & Min cov. & Area (m$^2$) \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table12_calibration_rules", body)


def table16():
    """Robustness over seeds: official tests (E9) and EETAC (E11)."""
    e9 = pd.read_csv(SRC / "E9_multiseed_official.csv")
    rows = []
    for n in ORDER:
        w = e9[e9.dataset == n].pivot_table(index="seed", columns="ratio", values="paper_metric_m")
        p9 = e9[e9.dataset == n].pivot_table(index="seed", columns="ratio", values="p90_m")
        impr = 100 * (w[1.0] - w[0.7]) / w[1.0]
        rows.append(f"{NICE[n]} & official test & {impr.mean():.1f} $\\pm$ {impr.std():.1f} & "
                    f"{int((w[0.7] < w[1.0]).sum())}/{len(w)} & {int((p9[0.7] < p9[1.0]).sum())}/{len(p9)} & -- \\\\")
    rows.append(r"\midrule")
    e11 = pd.read_csv(SRC / "E11_eetac_multiseed.csv")
    pname = {"sample_cv": "sample-level CV", "rp_grouped_cv": "RP-grouped CV",
             "d1_to_d2": r"Day~1$\rightarrow$Day~2", "d2_to_d1": r"Day~2$\rightarrow$Day~1"}
    sname = {"Medium_4Google": "EETAC Medium", "High_4Google_3Linksys": "EETAC High"}
    for s in ("Medium_4Google", "High_4Google_3Linksys"):
        first = True
        for pr in ("sample_cv", "rp_grouped_cv", "d1_to_d2", "d2_to_d1"):
            g = e11[(e11.scenario == s) & (e11.protocol == pr)]
            impr = 100 * (g.rmse_rho1 - g.rmse_rho07) / g.rmse_rho1
            ci = g[g.seed == 0].iloc[0]
            rows.append(f"{sname[s] if first else ''} & {pname[pr]} & {impr.mean():.1f} $\\pm$ {impr.std():.1f} & "
                        f"{int((g.rmse_rho07 < g.rmse_rho1).sum())}/{len(g)} & -- & "
                        f"[{ci.diff_ci95_low_m:+.3f}, {ci.diff_ci95_high_m:+.3f}] \\\\")
            first = False
    body = r"""\begin{table*}[!t]
\centering
\caption{Robustness of the comparison $\rho=0.7$ versus $\rho=1$ over ten seeds (0--9). Improvement: relative error reduction (paper-compatible metric for the benchmark, 2D RMSE for EETAC), mean $\pm$ standard deviation over seeds. Wins: seeds in which $\rho=0.7$ has the lower error (or lower $P_{90}$). CI: 95\% paired RP-clustered bootstrap interval of the 2D RMSE difference ($\rho=0.7$ minus $\rho=1$, m) for seed 0; for the official tests, see Table~\ref{tab:original_accuracy}.}
\label{tab:robustness}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
Dataset & Protocol & Improvement (\%) & Wins (error) & Wins ($P_{90}$) & 95\% CI of difference (m) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table16_robustness", body)


def table12_e4_unused():
    """Calibration rules (E4), circle score, over random RP splits."""
    d = pd.read_csv(SRC / "E4_calibration.csv")
    d = d[d.score == "circle"].copy()
    d["under"] = d.coverage < d.confidence
    rows = []
    for n in ORDER:
        g = d[d.dataset == n]
        G = int(g.cal_rps.iloc[0])
        nsp = g.split.nunique()
        first = True
        for rule in ("scan-level", "RP-corrected"):
            cells = []
            for conf in (0.90, 0.95):
                h = g[(g.rule == rule) & np.isclose(g.confidence, conf)]
                cells.append(f"{h.coverage.mean():.3f} & {h.coverage.min():.3f} & "
                             f"{100 * h.under.mean():.0f} & {h.mean_area_m2.mean():.2f}")
            lead = f"{NICE[n]} ($G={G}$, {nsp} splits)" if first else ""
            rows.append(f"{lead} & {rule} & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = r"""\begin{table*}[!t]
\centering
\caption{Calibration of circular split-conformal regions when the $\sim$120 scans of each RP are correlated. Scan-level: finite-sample level computed with $n=$ number of calibration scans (standard practice). RP-corrected: same pooled scores, level computed with $n=G$ calibration RPs. Statistics over repeated random RP-level fit/calibration splits ($25\%$ of training RPs for calibration); coverage measured on the official test set. Under: percentage of splits whose coverage falls below the nominal level.}
\label{tab:calibration_rules}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{$1-\alpha=0.90$} & \multicolumn{4}{c}{$1-\alpha=0.95$} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Dataset & Rule & Mean cov. & Min cov. & Under (\%) & Area (m$^2$) & Mean cov. & Min cov. & Under (\%) & Area (m$^2$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table12_calibration_rules", body)


def table13():
    """Compact forests (E6)."""
    d = pd.read_csv(SRC / "E6_model_size.csv")
    d["max_depth"] = d.max_depth.fillna("None").astype(str)
    configs = [(1.0, 300, 1, "None"), (0.7, 300, 1, "None"), (0.7, 100, 1, "None"),
               (0.7, 50, 1, "None"), (0.7, 50, 5, "None")]
    rows = []
    for n in ORDER:
        g = d[d.dataset == n]
        first = True
        for r, t, leaf, depth in configs:
            h = g[np.isclose(g.ratio, r) & (g.n_trees == t) & (g.min_samples_leaf == leaf)
                  & (g.max_depth == depth)].iloc[0]
            lead = NICE[n] if first else ""
            rows.append(f"{lead} & {r:.1f} & {t} & {leaf} & {h.size_mb:.2f} & "
                        f"{h.infer_us_per_scan_1thread:.1f} & {h.train_s:.2f} & "
                        f"{h.paper_metric_m:.3f} & {h.rmse2d_m:.3f} \\\\")
            first = False
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = r"""\begin{table*}[!t]
\centering
\caption{Compact forests. Serialized model size (Python pickle), single-thread inference time per scan, training time (20 threads, Intel Core i5-13500HX), and official-test accuracy. Error: paper-compatible metric (m); RMSE: true 2D RMSE (m).}
\label{tab:compact_models}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
Dataset & $\rho$ & Trees & Min. leaf & Size (MB) & Inference ($\mu$s/scan) & Train (s) & Error (m) & 2D RMSE (m) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table13_compact_models", body)


def table15():
    """Best ratio and gain over rho=1 per density, RF (E1) vs ExtraTrees (E7)."""
    if not (SRC / "E7_density_ratio_extratrees.csv").exists():
        return
    models = {"RF": pd.read_csv(SRC / "E1_density_ratio.csv"),
              "ET": pd.read_csv(SRC / "E7_density_ratio_extratrees.csv")}
    rows = []
    for name in ORDER:
        first = True
        for frac in (1.0, 0.75, 0.5, 0.25):
            cells = []
            rps = None
            for mod, d in models.items():
                g = d[(d.dataset == name) & np.isclose(d.rp_fraction, frac)]
                rps = int(g.train_rps.iloc[0])
                s = g.groupby("ratio").paper_metric_m.mean()
                best = s.idxmin()
                gain = 100 * (s.loc[1.0] - s.min()) / s.loc[1.0]
                cells.append(f"{best:.2f} & {s.min():.3f} & {gain:.1f}")
            lead = NICE[name] if first else ""
            rows.append(f"{lead} & {int(frac * 100)}\\% ({rps}) & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\midrule")
    nseed = models["RF"].seed.nunique()
    body = r"""\begin{table*}[!t]
\centering
\caption{Density dependence of the optimal subspace ratio for Random Forests and extremely randomized trees (official test, paper-compatible metric, mean over NSEED random RP subsets; both models use the same thinned radio maps). Best $\rho$: ratio with the lowest mean error in $\{0.2,0.35,0.5,0.7,0.85,1.0\}$. Gain: reduction of the best ratio relative to $\rho=1$.}
\label{tab:density_rf_et}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{Random Forest} & \multicolumn{3}{c}{Extremely randomized trees} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Dataset & Training map & Best $\rho$ & Error (m) & Gain (\%) & Best $\rho$ & Error (m) & Gain (\%) \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table15_density_rf_et", body.replace("NSEED", str(nseed)))


if __name__ == "__main__":
    fig13()
    fig14()
    fig15()
    table10()
    table11()
    table12()
    table13()
    table15()
    table16()
    print("figures written to", DST)
    print("tables written to", TAB)
