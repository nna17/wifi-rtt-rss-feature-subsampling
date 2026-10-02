"""
make_supplementary.py
=====================

Writes paper/supplementary/supplementary_tables.tex from the
audit CSVs.  Run from the project root:

    python scripts/extensions/make_supplementary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
AUD = ROOT / "audit_results"
EXT = AUD / "extensions"
DST = ROOT / "paper" / "supplementary"
NICE = {"building": "Building", "apartment": "Apartment", "office": "Office",
        "Medium_4Google": "Medium (4 APs)", "High_4Google_3Linksys": "High (7 APs)"}


def table(caption, label, colspec, header, rows):
    return ("\\begin{table}[!ht]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\footnotesize\n\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def s1_frame_audit():
    d = pd.read_csv(ROOT / "eetac_external_validation_results" / "day_frame_alignment_audit.csv").iloc[0]
    rows = [f"Identity ($y\\mapsto y$) & {d.identity_matches} \\\\",
            f"Mirror ($y\\mapsto 8-y$) & {d.mirror_y_matches} \\\\",
            f"Uninformative (central row $y=4$~m) & {d.uninformative_centre_row_rps} \\\\"]
    return table("EETAC acquisition-day frame audit. For each of the "
                 f"{d.n_rps} Day~1 RPs, the Day~2 RP with the nearest median Google-AP RTT fingerprint is found; "
                 "the table counts how often it lies at the position predicted by each coordinate mapping. "
                 f"Selected mapping: {d.selected.replace('_', ' ')}.",
                 "tab:s_frame_audit", "lc", "Mapping of Day~2 coordinates & Matching RPs (of 25)", rows)


def s2_alignment_effect():
    d = pd.read_csv(AUD / "C_eetac_flip.csv")
    cols = ["sample_cv_rmse2d", "rp_grouped_cv_rmse2d", "d1_to_d2_rmse2d", "d2_to_d1_rmse2d"]
    g = d.groupby(["y_flip_day2", "scenario", "max_features"])[cols].mean()
    rows = []
    for flip in (False, True):
        for scen in ("Medium_4Google", "High_4Google_3Linksys"):
            for mf in (1.0, 0.7):
                v = g.loc[(flip, scen, mf)]
                rows.append(f"{'aligned' if flip else 'not aligned'} & {NICE[scen]} & {mf:.1f} & "
                            + " & ".join(f"{x:.3f}" for x in v) + " \\\\")
        if not flip:
            rows.append("\\midrule")
    return table("EETAC 2D RMSE (m) with and without acquisition-day frame alignment, mean of three seeds (0--2). "
                 "Without alignment, pooled and cross-day experiments mix two physical locations under one label.",
                 "tab:s_alignment", "lllcccc",
                 "Frames & Scenario & $\\rho$ & Sample CV & RP-grouped CV & D1$\\rightarrow$D2 & D2$\\rightarrow$D1", rows)


def s3_eetac_ratio():
    d = pd.read_csv(EXT / "E5_eetac_ratio.csv")
    g = d.groupby(["scenario", "ratio"])[["sample_cv", "rp_grouped_cv", "d1_to_d2", "d2_to_d1"]].mean()
    rows = []
    for scen in ("Medium_4Google", "High_4Google_3Linksys"):
        sub = g.loc[scen]
        best = sub.idxmin()
        for r, v in sub.iterrows():
            cells = []
            for c in sub.columns:
                s = f"{v[c]:.3f}"
                cells.append(f"\\textbf{{{s}}}" if np.isclose(r, best[c]) else s)
            rows.append(f"{NICE[scen] if np.isclose(r, sub.index[0]) else ''} & {r:.2f} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    return table("EETAC ratio sweep (frame-aligned), 2D RMSE (m), mean of three seeds. Best ratio per column in bold.",
                 "tab:s_eetac_ratio", "lccccc",
                 "Scenario & $\\rho$ & Sample CV & RP-grouped CV & D1$\\rightarrow$D2 & D2$\\rightarrow$D1", rows[:-1])


def s4_density():
    d = pd.read_csv(EXT / "E1_density_ratio.csv")
    rows = []
    for name in ("building", "apartment", "office"):
        g = d[d.dataset == name]
        first = True
        for frac in (1.0, 0.75, 0.5, 0.25):
            h = g[np.isclose(g.rp_fraction, frac)].groupby("ratio").paper_metric_m.agg(["mean", "std"])
            best = h["mean"].idxmin()
            cells = []
            for r, v in h.iterrows():
                s = f"{v['mean']:.3f}$\\pm${v['std']:.3f}"
                cells.append(f"\\textbf{{{s}}}" if np.isclose(r, best) else s)
            rps = int(g[np.isclose(g.rp_fraction, frac)].train_rps.iloc[0])
            rows.append(f"{NICE[name] if first else ''} & {int(frac * 100)}\\% ({rps}) & " + " & ".join(cells) + " \\\\")
            first = False
        rows.append("\\midrule")
    ratios = sorted(d.ratio.unique())
    nseed = d.seed.nunique()
    return ("\\begin{table*}[!ht]\n\\centering\n\\caption{Official-test paper-compatible error (m, mean $\\pm$ std over "
            + str(nseed) + " random RP subsets) "
            "versus subspace ratio for thinned training radio maps (fraction and number of training RPs). Best ratio in bold.}\n"
            "\\label{tab:s_density}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{ll" + "c" * len(ratios) + "}\n\\toprule\n"
            "Dataset & Map & " + " & ".join(f"$\\rho={r}$" for r in ratios) + " \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def s5_screening():
    small, big = AUD / "small", AUD / "building"
    n2 = pd.concat([pd.read_csv(small / "N2_ap_subspace.csv"), pd.read_csv(big / "N2_ap_subspace.csv")])
    n34 = pd.concat([pd.read_csv(small / "N34_geometry.csv"), pd.read_csv(big / "N34_geometry.csv")])
    rows = []
    label = {"RF-node-F0.7": "Per-split subsampling, $\\rho=0.7$ (reference)",
             "tree-feature-0.7": "Per-tree feature subspace, $70\\%$ of features",
             "tree-AP-0.7": "Per-tree AP subspace, $70\\%$ of APs (RTT+RSS kept together)",
             "tree-AP-0.5": "Per-tree AP subspace, $50\\%$ of APs"}
    m2 = n2.groupby(["model", "dataset"])[["test_paper_metric_m", "cv_paper_metric_m"]].mean()
    for mod, lab in label.items():
        cells = " & ".join(f"{m2.loc[(mod, n), 'test_paper_metric_m']:.3f} / {m2.loc[(mod, n), 'cv_paper_metric_m']:.3f}"
                           for n in ("building", "apartment", "office"))
        rows.append(f"{lab} & {cells} \\\\")
    rows.append("\\midrule")
    vlabel = {"base": "RF $\\rho=0.7$, RTT+RSS only (reference)",
              "geo": "+ multilateration features (position, \\#APs, residual)",
              "geo+diff": "+ multilateration + RTT/RSS differences to strongest AP",
              "residual": "Multilateration + RF residual correction"}
    m3 = n34[np.isclose(n34.max_features, 0.7)].groupby(["variant", "dataset"])[["test_paper_metric_m", "cv_paper_metric_m"]].mean()
    for v, lab in vlabel.items():
        cells = " & ".join(f"{m3.loc[(v, n), 'test_paper_metric_m']:.3f} / {m3.loc[(v, n), 'cv_paper_metric_m']:.3f}"
                           for n in ("building", "apartment", "office"))
        rows.append(f"{lab} & {cells} \\\\")
    return ("\\begin{table*}[!ht]\n\\centering\n\\caption{Screening experiments not retained in the main paper "
            "(paper-compatible metric in m, official test / pooled RP-grouped CV; Office and Apartment: mean of three seeds, "
            "Building: one seed). Per-tree subspaces are worse than per-split subsampling with or without pairing an AP's RTT and RSS. "
            "Geometry features from RTT multilateration with the published AP coordinates help the small three- and four-AP environments "
            "under RP-grouped CV but not Building; the residual variant is unstable because multilateration diverges for some scans.}\n"
            "\\label{tab:s_screening}\n\\footnotesize\n\\begin{tabular}{lccc}\n\\toprule\n"
            "Variant & Building & Apartment & Office \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def s6_local_uncertainty():
    small, big = AUD / "small", AUD / "building"
    d = pd.concat([pd.read_csv(small / "N5_local_uncertainty.csv"), pd.read_csv(big / "N5_local_uncertainty.csv")])
    g = d.groupby(["dataset", "confidence", "method"])[["coverage", "mean_area_m2"]].mean()
    rows = []
    for n in ("building", "apartment", "office"):
        for conf in (0.90, 0.95):
            cells = " & ".join(f"{g.loc[(n, conf, m), 'coverage']:.3f} / {g.loc[(n, conf, m), 'mean_area_m2']:.2f}"
                               for m in ("global-circle", "global-ellipse", "local-circle", "local-ellipse"))
            rows.append(f"{NICE[n] if conf == 0.90 else ''} & {int(conf * 100)}\\% & {cells} \\\\")
    return ("\\begin{table*}[!ht]\n\\centering\n\\caption{Screening of locally adaptive conformal regions (scan-level split conformal; "
            "coverage / mean area in m$^2$). Local regions normalize the residual by the spread of the 300 tree predictions "
            "(plus a fixed floor equal to the median spread). Mean over three seeds (Building: one).}\n"
            "\\label{tab:s_local}\n\\footnotesize\n\\begin{tabular}{llcccc}\n\\toprule\n"
            "Dataset & $1-\\alpha$ & Global circle & Global ellipse & Local circle & Local ellipse \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


MAIN = ROOT / "paper_results" / "tables"


def s7_efficiency():
    d = pd.read_csv(MAIN / "table07_efficiency.csv")
    names = {"Building": "Building", "Office": "Office", "Apartment": "Apartment",
             "EETAC Medium_4Google": "EETAC Medium (4 APs)",
             "EETAC High_4Google_3Linksys": "EETAC High (7 APs)"}

    def f(v, fmt):
        return "--" if pd.isna(v) else fmt.format(v)

    rows = []
    for key, label in names.items():
        g = d[d["Dataset / scenario"] == key]
        if g.empty:
            continue
        b = g[g.Model == "RF300-F1.0"].iloc[0]
        p = g[g.Model == "RF300-F0.7"].iloc[0]
        rows.append(f"{label} & {b['Accuracy metric']:.3f} & {p['Accuracy metric']:.3f} & "
                    f"{b['Train time (s)']:.2f} & {p['Train time (s)']:.2f} & "
                    f"{f(b['Inference (us/sample)'], '{:.2f}')} & {f(p['Inference (us/sample)'], '{:.2f}')} & "
                    f"{f(b['Model size (MB)'], '{:.2f}')} & {f(p['Model size (MB)'], '{:.2f}')} \\\\")
    return ("\\begin{table*}[!ht]\n\\centering\n\\caption{Accuracy and cost of the 300-tree models (seed 42, 20 threads). "
            "Error: paper-compatible metric for the original benchmark, 2D RMSE (sample-level CV) for EETAC. "
            "Wall-clock times depend on system load; compact forests are analyzed in the paper.}\n"
            "\\label{tab:s_efficiency}\n\\footnotesize\n\\begin{tabular}{lcccccccc}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Error (m)} & \\multicolumn{2}{c}{Train (s)} & \\multicolumn{2}{c}{Inference ($\\mu$s/scan)} & \\multicolumn{2}{c}{Size (MB)} \\\\\n"
            "Dataset & $\\rho=1$ & $\\rho=0.7$ & $\\rho=1$ & $\\rho=0.7$ & $\\rho=1$ & $\\rho=0.7$ & $\\rho=1$ & $\\rho=0.7$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def s8_uncertainty():
    d = pd.read_csv(MAIN / "table09_uncertainty.csv")
    short = {"Strict split conformal": "Split conformal, scan-level",
             "Strict split conformal (RP-corrected)": "Split conformal, RP-count",
             "Grouped-OOF empirical": "Grouped-OOF empirical"}
    rows = []
    for ds in ("Building", "Office", "Apartment"):
        g = d[d.Dataset == ds]
        G = g["Calibration RPs"].dropna()
        first = True
        for prot in short:
            cells = []
            for conf in (0.90, 0.95):
                for region in ("Circle", "Ellipse"):
                    r = g[(g.Protocol == prot) & np.isclose(g.Confidence, conf) & (g.Region == region)].iloc[0]
                    cells.append(f"{r.Coverage:.3f} & {r['Area (m^2)']:.2f}")
            lead = f"{ds} ($G={int(G.iloc[0])}$)" if first else ""
            rows.append(f"{lead} & {short[prot]} & " + " & ".join(cells) + " \\\\")
            first = False
        rows.append("\\midrule")
    return ("\\begin{table*}[!ht]\n\\centering\n\\caption{Circular and covariance-aware elliptical regions in a single calibration split "
            "(seed 42): empirical coverage (Cov.) and area (m$^2$). $G$: number of calibration RPs. RP-count: the conservative "
            "level computed with $n=G$. The variability over calibration splits, and the valid group-conformal rule, are reported in the paper.}\n"
            "\\label{tab:s_uncertainty}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{llcccccccc}\n\\toprule\n"
            " & & \\multicolumn{4}{c}{$1-\\alpha=0.90$} & \\multicolumn{4}{c}{$1-\\alpha=0.95$} \\\\\n"
            " & & \\multicolumn{2}{c}{Circle} & \\multicolumn{2}{c}{Ellipse} & \\multicolumn{2}{c}{Circle} & \\multicolumn{2}{c}{Ellipse} \\\\\n"
            "Dataset & Calibration & Cov. & Area & Cov. & Area & Cov. & Area & Cov. & Area \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


if __name__ == "__main__":
    parts = [s1_frame_audit(), s2_alignment_effect(), s3_eetac_ratio(), s4_density(), s5_screening(),
             s6_local_uncertainty(), s7_efficiency(), s8_uncertainty()]
    (DST / "supplementary_tables.tex").write_text(
        "% Generated by scripts/extensions/make_supplementary.py -- do not edit by hand.\n\n" + "\n".join(parts), encoding="utf-8")
    print("written", DST / "supplementary_tables.tex")
