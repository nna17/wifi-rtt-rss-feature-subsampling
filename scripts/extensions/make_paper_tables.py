"""
make_paper_tables.py
====================

Writes the manuscript versions of the main-comparison tables (III-IX) from
the CSVs produced by run_all_paper_experiments.py, in the booktabs layout
used by the paper.  Run from the project root:

    python scripts/extensions/make_paper_tables.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / "paper_results" / "tables"
DST = ROOT / "paper" / "tables"

SCEN = {"Medium_4Google": "Medium (4 APs)", "High_4Google_3Linksys": "High (7 APs)"}


def f3(v):
    return f"{v:.3f}"


def pair(a, b, fmt=f3):
    """Bold the smaller of two values (lower is better)."""
    sa, sb = fmt(a), fmt(b)
    if np.isclose(a, b):
        return sa, sb
    return (rf"\textbf{{{sa}}}", sb) if a < b else (sa, rf"\textbf{{{sb}}}")


def write(stem, text):
    (DST / f"{stem}.tex").write_text(text, encoding="utf-8")


def table03():
    d = pd.read_csv(SRC / "table03_original_accuracy.csv")
    rows = []
    for _, r in d.iterrows():
        e1, e7 = pair(r["Reproduced F1.0 (m)"], r["Proposed F0.7 (m)"])
        m1, m7 = pair(r["2D RMSE F1.0 (m)"], r["2D RMSE F0.7 (m)"])
        p1, p7 = pair(r["P90 F1.0 (m)"], r["P90 F0.7 (m)"])
        ci = f"[{r['Diff 95% CI low (m)']:+.3f}, {r['Diff 95% CI high (m)']:+.3f}]"
        rows.append(f"{r['Dataset']} & {r['Published metric (m)']:.3f} & {e1} & {e7} & "
                    f"{r['Improvement vs F1.0 (%)']:.2f} & {ci} & {m1} & {m7} & {p1} & {p7} \\\\")
    write("table03_original_accuracy", r"""\begin{table*}[!t]
\centering
\caption{Point-localization performance on the original hybrid RTT--RSS benchmark (official RP-disjoint test sets). The paper-compatible metric is defined in Eq.~\eqref{eq:paper_metric}. 95\% CI: paired cluster bootstrap over test RPs (1000 replicates) of the F0.7$-$F1.0 difference in the paper-compatible metric; an interval excluding zero indicates a reliable improvement.}
\label{tab:original_accuracy}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{5}{c}{Paper-compatible metric (m)} & \multicolumn{2}{c}{2D RMSE (m)} & \multicolumn{2}{c}{$P_{90}$ (m)} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Dataset & Published & F1.0 & F0.7 & Impr. (\%) & 95\% CI of diff. & F1.0 & F0.7 & F1.0 & F0.7 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def table04():
    d = pd.read_csv(SRC / "table04_grouped_cv.csv")
    rows = []
    for _, r in d.iterrows():
        a = pair(r["F1.0 paper metric (m)"], r["F0.7 paper metric (m)"])
        b = pair(r["F1.0 2D RMSE (m)"], r["F0.7 2D RMSE (m)"])
        c = pair(r["F1.0 P90 (m)"], r["F0.7 P90 (m)"])
        e = pair(r["F1.0 P95 (m)"], r["F0.7 P95 (m)"])
        rows.append(f"{r['Dataset']} & {a[0]} & {a[1]} & {r['Improvement (%)']:.2f} & "
                    f"{b[0]} & {b[1]} & {c[0]} & {c[1]} & {e[0]} & {e[1]} \\\\")
    write("table04_grouped_cv", r"""\begin{table*}[!t]
\centering
\caption{Five-fold RP-grouped validation on the official training RPs of the original benchmark. All metrics are computed on the pooled out-of-fold predictions. Each fold model is trained on about $80\%$ of the training RPs, i.e., on a sparser radio map than the official-test models.}
\label{tab:grouped_cv}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{3}{c}{Paper-compatible metric (m)} & \multicolumn{2}{c}{2D RMSE (m)} & \multicolumn{2}{c}{$P_{90}$ (m)} & \multicolumn{2}{c}{$P_{95}$ (m)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Dataset & F1.0 & F0.7 & Impr. (\%) & F1.0 & F0.7 & F1.0 & F0.7 & F1.0 & F0.7 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def table05():
    d = pd.read_csv(SRC / "table05_eetac_external_validation.csv")
    rows = []
    for scen, g in d.groupby("Scenario", sort=False):
        first = True
        for _, r in g.iterrows():
            a = pair(r["F1.0 2D RMSE (m)"], r["F0.7 2D RMSE (m)"])
            b = pair(r["F1.0 P90 (m)"], r["F0.7 P90 (m)"])
            c = pair(r["F1.0 P95 (m)"], r["F0.7 P95 (m)"])
            lead = SCEN[scen] if first else ""
            rows.append(f"{lead} & {r['Protocol']} & {a[0]} & {a[1]} & {r['RMSE improvement (%)']:.2f} & "
                        f"{b[0]} & {b[1]} & {c[0]} & {c[1]} \\\\")
            first = False
        rows.append(r"\midrule")
    write("table05_eetac_external_validation", r"""\begin{table*}[!t]
\centering
\caption{External validation on the independent EETAC auditorium dataset (both acquisition days pooled after frame alignment; seed 42). Medium: four Google APs; High: four Google and three Linksys/Belkin APs. Sample-level CV is stratified by RP; RP-grouped CV holds out complete RPs of the $5\times5$ grid.}
\label{tab:eetac_external}
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llccccccc}
\toprule
 & & \multicolumn{3}{c}{2D RMSE (m)} & \multicolumn{2}{c}{$P_{90}$ (m)} & \multicolumn{2}{c}{$P_{95}$ (m)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Scenario & Protocol & F1.0 & F0.7 & Impr. (\%) & F1.0 & F0.7 & F1.0 & F0.7 \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def table06():
    d = pd.read_csv(SRC / "table06_cross_day.csv")
    rows = []
    for scen, g in d.groupby("Scenario", sort=False):
        first = True
        for _, r in g.iterrows():
            a = pair(r["F1.0 2D RMSE (m)"], r["F0.7 2D RMSE (m)"])
            b = pair(r["F1.0 P90 (m)"], r["F0.7 P90 (m)"])
            direction = r["Direction"].replace("Day 1 -> Day 2", r"Day~1$\rightarrow$Day~2") \
                                      .replace("Day 2 -> Day 1", r"Day~2$\rightarrow$Day~1")
            lead = SCEN[scen] if first else ""
            rows.append(f"{lead} & {direction} & {a[0]} & {a[1]} & {r['RMSE improvement (%)']:.2f} & "
                        f"{b[0]} & {b[1]} & {r['P90 improvement (%)']:.2f} \\\\")
            first = False
        rows.append(r"\midrule")
    write("table06_cross_day", r"""\begin{table*}[!t]
\centering
\caption{Cross-day EETAC temporal generalization after acquisition-day frame alignment (seed 42). Each model is trained on all scans of one day and tested on all scans of the other. Negative improvements indicate that F0.7 is worse.}
\label{tab:cross_day}
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{2D RMSE (m)} & \multicolumn{3}{c}{$P_{90}$ (m)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Scenario & Direction & F1.0 & F0.7 & Impr. (\%) & F1.0 & F0.7 & Impr. (\%) \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def table07():
    d = pd.read_csv(SRC / "table07_efficiency.csv")
    names = {"Building": "Building", "Office": "Office", "Apartment": "Apartment",
             "EETAC Medium_4Google": "EETAC Medium (4 APs)",
             "EETAC High_4Google_3Linksys": "EETAC High (7 APs)"}
    rows = []
    for key, label in names.items():
        g = d[d["Dataset / scenario"] == key]
        if g.empty:
            continue
        b = g[g.Model == "RF300-F1.0"].iloc[0]
        p = g[g.Model == "RF300-F0.7"].iloc[0]
        e = pair(b["Accuracy metric"], p["Accuracy metric"])

        def num(v, fmt):
            return "--" if pd.isna(v) else fmt.format(v)

        rows.append(f"{label} & {e[0]} & {e[1]} & {b['Train time (s)']:.2f} & {p['Train time (s)']:.2f} & "
                    f"{b['Prediction time (s)']:.3f} & {p['Prediction time (s)']:.3f} & "
                    f"{num(b['Inference (us/sample)'], '{:.2f}')} & {num(p['Inference (us/sample)'], '{:.2f}')} & "
                    f"{num(b['Model size (MB)'], '{:.2f}')} & {num(p['Model size (MB)'], '{:.2f}')} \\\\")
    write("table07_efficiency", r"""\begin{table*}[!t]
\centering
\caption{Accuracy and computational measurements of the 300-tree models (20 threads). The error is the paper-compatible metric for the original benchmark and the 2D RMSE for EETAC (sample-level CV); per-sample inference time and model size were not recorded for EETAC. Compact forests are analyzed in Table~\ref{tab:compact_models}.}
\label{tab:efficiency}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lcccccccccc}
\toprule
 & \multicolumn{2}{c}{Error (m)} & \multicolumn{2}{c}{Train time (s)} & \multicolumn{2}{c}{Prediction time (s)} & \multicolumn{2}{c}{Inference ($\mu$s/sample)} & \multicolumn{2}{c}{Model size (MB)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Dataset & F1.0 & F0.7 & F1.0 & F0.7 & F1.0 & F0.7 & F1.0 & F0.7 & F1.0 & F0.7 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def table09():
    d = pd.read_csv(SRC / "table09_uncertainty.csv")
    order = ["Strict split conformal", "Strict split conformal (RP-corrected)", "Grouped-OOF empirical"]
    short = {"Strict split conformal": "Split conformal (scan-level)",
             "Strict split conformal (RP-corrected)": "Split conformal (RP-count)",
             "Grouped-OOF empirical": "Grouped-OOF empirical"}
    rows = []
    for ds in ("Building", "Office", "Apartment"):
        g = d[d.Dataset == ds]
        G = g["Calibration RPs"].dropna()
        first = True
        for prot in order:
            cells = []
            for conf in (0.90, 0.95):
                for region in ("Circle", "Ellipse"):
                    r = g[(g.Protocol == prot) & np.isclose(g.Confidence, conf) & (g.Region == region)].iloc[0]
                    cells.append(f"{r.Coverage:.3f} & {r['Area (m^2)']:.2f}")
            lead = f"{ds} ($G={int(G.iloc[0])}$)" if first else ""
            rows.append(f"{lead} & {short[prot]} & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\midrule")
    write("table09_uncertainty", r"""\begin{table*}[!t]
\centering
\caption{Circular and covariance-aware elliptical uncertainty regions: empirical coverage (Cov.) on the official test set and region area (m$^2$) at nominal $90\%$ and $95\%$ levels (single calibration split, seed 42). $G$: number of calibration RPs. Scan-level and RP-count rows use the same predictor and scores and differ only in the finite-sample level (Section~\ref{subsubsec:rp_corrected}). The variability of these results over calibration splits is reported in Table~\ref{tab:calibration_rules}.}
\label{tab:uncertainty}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{$1-\alpha=0.90$} & \multicolumn{4}{c}{$1-\alpha=0.95$} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
 & & \multicolumn{2}{c}{Circle} & \multicolumn{2}{c}{Ellipse} & \multicolumn{2}{c}{Circle} & \multicolumn{2}{c}{Ellipse} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Dataset & Protocol & Cov. & Area & Cov. & Area & Cov. & Area & Cov. & Area \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


if __name__ == "__main__":
    for fn in (table03, table04, table05, table06, table07, table09):
        fn()
    print("written:", DST)
