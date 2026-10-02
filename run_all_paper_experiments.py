"""
run_all_paper_experiments.py
============================

Final experiment orchestrator for the paper:

    Efficient Hybrid WiFi RTT-RSS Indoor Localization with
    Random-Subspace Forests, Reference-Point-Aware Validation,
    and Uncertainty Quantification

WHAT THIS SCRIPT DOES
---------------------
1. Runs the validated original-dataset experiment engine:
       paper_all_contributions_wifi_rtt_rss.py

2. Runs the corrected EETAC external-validation engine:
       eetac_external_validation_subspace_rf_v2.py

3. Reads their raw CSV outputs.

4. Generates the PREDEFINED paper outputs only:

   TABLES
   ------
   table01_dataset_summary
   table02_model_configuration
   table03_original_accuracy
   table04_grouped_cv
   table05_eetac_external_validation
   table06_cross_day
   table07_efficiency
   table08_data_utilization
   table09_uncertainty

   FIGURES
   -------
   fig06_original_accuracy
   fig07_error_cdf
   fig08_grouped_cv
   fig09_eetac_external_validation
   fig10_accuracy_efficiency
   fig11_cross_day
   fig12_uncertainty_area_coverage

5. Copies all numerical source CSVs into paper_results/raw/.

6. Captures complete experiment logs in paper_results/logs/.

IMPORTANT SCIENTIFIC RULES
--------------------------
- RF300-F1.0 is the full-feature baseline.
- RF300-F0.7 is the frozen proposed configuration.
- F=0.7 is NOT retuned on EETAC.
- Original-dataset published comparisons use the paper-compatible metric.
- EETAC comparisons use true 2D RMSE in metres.
- Sample-level and RP-grouped EETAC results are reported separately.
- Strict split-conformal and grouped-OOF empirical uncertainty are reported
  separately.  The OOF variant is NOT labelled as ordinary split conformal.
- EETAC Low/2-AP scenarios are NOT used in the main paper tables because the
  public files do not unambiguously encode the exact opposite-corner AP labels.
- Main EETAC tables emphasize:
      Medium_4Google
      High_4Google_3Linksys
  when available.

EXPECTED PROJECT LAYOUT
-----------------------
project/
│
├── run_all_paper_experiments.py
├── dataset/
│   ├── database_building_train.csv
│   ├── database_building_test.csv
│   ├── database_office_train.csv
│   ├── database_office_test.csv
│   ├── database_apartment_train.csv
│   ├── database_apartment_test.csv
│   ├── EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 1.csv
│   ├── EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 1.csv
│   ├── EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 2.csv
│   └── EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 2.csv
│
└── scripts/
    ├── original_dataset/
    │   └── paper_all_contributions_wifi_rtt_rss.py
    └── eetac/
        └── eetac_external_validation_subspace_rf_v2.py

OUTPUT
------
paper_results/
├── tables/
├── figures/
├── raw/
└── logs/

INTERMEDIATE VALIDATED OUTPUTS
------------------------------
paper_results_all_contributions/
eetac_external_validation_results/

RUN
---
    python run_all_paper_experiments.py

Optional:
    python run_all_paper_experiments.py --build-only
    python run_all_paper_experiments.py --project-root "C:\\path\\to\\project"

Dependencies:
    numpy
    pandas
    matplotlib
    scikit-learn

The helper scripts remain computation engines, but THIS is the only file you
need to run. It launches both helpers, protects their output folders, collects
their data, and generates the final manuscript tables/figures automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONSTANTS: FROZEN PAPER PROTOCOL
# =============================================================================

SEED = 42
N_TREES = 300
BASELINE_MAX_FEATURES = 1.0
PROPOSED_MAX_FEATURES = 0.7
CV_FOLDS = 5
CALIBRATION_FRACTION = 0.25
CONFIDENCE_LEVELS = (0.90, 0.95)

ORIGINAL_DATASETS = ("building", "office", "apartment")

PUBLISHED_ORIGINAL = {
    "building": 0.60,
    "office": 0.38,
    "apartment": 0.59,
}

SOURCE_REPORTED_TOTAL_RPS = {
    "building": 642,
    "office": 37,
    "apartment": 110,
}

SOURCE_AREA = {
    "building": "92 x 15 m^2",
    "office": "5.5 x 4.5 m^2",
    "apartment": "7.7 x 9.4 m^2",
}

SOURCE_GRID_M = {
    "building": 0.600,
    "office": 0.455,
    "apartment": 0.480,
}

EETAC_MAIN_SCENARIOS = (
    "Medium_4Google",
    "High_4Google_3Linksys",
)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all frozen experiments and generate paper outputs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project directory containing dataset/. Defaults to current directory.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help=(
            "Do not rerun the two computation engines. Rebuild manuscript "
            "tables/figures from existing intermediate results."
        ),
    )
    parser.add_argument(
        "--keep-paper-results",
        action="store_true",
        help="Do not delete/recreate paper_results/ before building outputs.",
    )
    return parser.parse_args()


def resolve_project_root(arg: Path | None) -> Path:
    candidates = []
    if arg is not None:
        candidates.append(arg.expanduser().resolve())

    candidates.extend([
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ])

    for candidate in candidates:
        if (candidate / "dataset").is_dir():
            return candidate

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not locate project root containing dataset/.\n"
        f"Checked:\n{checked}\n"
        "Run with --project-root if necessary."
    )


def find_helper(project_root: Path, filename: str, kind: str) -> Path:
    here = Path(__file__).resolve().parent

    candidates = [
        project_root / filename,
        project_root / "scripts" / filename,
        project_root / "scripts" / kind / filename,
        here / filename,
        here / "scripts" / filename,
        here / "scripts" / kind / filename,
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Missing helper script: {filename}\nChecked:\n{checked}"
    )


def ensure_dataset_files(project_root: Path) -> None:
    required = [
        "database_building_train.csv",
        "database_building_test.csv",
        "database_office_train.csv",
        "database_office_test.csv",
        "database_apartment_train.csv",
        "database_apartment_test.csv",
        "EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 1.csv",
        "EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 1.csv",
        "EETAC Auditorium-C3-0-google-Pixel 3a-RSS-day 2.csv",
        "EETAC Auditorium-C3-0-google-Pixel 3a-RTT-day 2.csv",
    ]

    missing = [
        project_root / "dataset" / name
        for name in required
        if not (project_root / "dataset" / name).exists()
    ]

    if missing:
        formatted = "\n".join(f"  - {p.name}" for p in missing)
        raise FileNotFoundError(
            "The frozen experiment requires all ten dataset files.\n"
            f"Missing:\n{formatted}"
        )


def run_engine(
    script: Path,
    project_root: Path,
    log_path: Path,
) -> None:
    """
    Execute a helper experiment engine robustly.

    The helper file itself is NOT modified.  It is imported in a child Python
    process, its known save functions are wrapped so their output directories
    are recreated immediately before every write, and a lightweight watchdog
    keeps OUT_DIR/TABLE_DIR/PLOT_DIR present throughout long Windows runs.

    This specifically prevents failures such as:
        FileNotFoundError:
        paper_results_all_contributions\\plots\\...png
    """
    print("\n" + "=" * 100)
    print(f"RUNNING: {script.name}")
    print("=" * 100)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-create the two known helper output trees before launching either
    # engine.  The child wrapper also keeps them alive during execution.
    if script.name == "paper_all_contributions_wifi_rtt_rss.py":
        expected_dirs = [
            project_root / "paper_results_all_contributions",
            project_root / "paper_results_all_contributions" / "tables",
            project_root / "paper_results_all_contributions" / "plots",
        ]
    elif script.name == "eetac_external_validation_subspace_rf_v2.py":
        expected_dirs = [
            project_root / "eetac_external_validation_results",
            project_root / "eetac_external_validation_results" / "plots",
        ]
    else:
        expected_dirs = []

    for d in expected_dirs:
        d.mkdir(parents=True, exist_ok=True)

    wrapper_code = r"""
import importlib.util
import pathlib
import sys
import threading
import traceback

script_path = pathlib.Path(sys.argv[1]).resolve()

spec = importlib.util.spec_from_file_location("_paper_engine", script_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import experiment engine: {script_path}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def ensure_engine_dirs():
    for attr in ("OUT_DIR", "TABLE_DIR", "PLOT_DIR", "RESULTS_DIR"):
        value = getattr(mod, attr, None)
        if value is not None:
            pathlib.Path(value).mkdir(parents=True, exist_ok=True)

ensure_engine_dirs()

# Wrap known writer functions without changing the helper source file.
for fn_name in ("save_figure", "save_fig", "save_table"):
    original = getattr(mod, fn_name, None)
    if callable(original):
        def make_safe_writer(fn):
            def safe_writer(*args, **kwargs):
                ensure_engine_dirs()
                return fn(*args, **kwargs)
            return safe_writer
        setattr(mod, fn_name, make_safe_writer(original))

# Long experiments can run for minutes.  Keep expected output folders present
# even if Windows, sync software, or another process temporarily removes them.
stop_event = threading.Event()

def directory_watchdog():
    while not stop_event.wait(0.5):
        try:
            ensure_engine_dirs()
        except Exception:
            pass

watchdog = threading.Thread(target=directory_watchdog, daemon=True)
watchdog.start()

try:
    main_fn = getattr(mod, "main", None)
    if not callable(main_fn):
        raise RuntimeError(f"No callable main() found in {script_path.name}")
    main_fn()
finally:
    stop_event.set()
    watchdog.join(timeout=1.0)
"""

    t0 = time.perf_counter()

    proc = subprocess.run(
        [sys.executable, "-c", wrapper_code, str(script)],
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    elapsed = time.perf_counter() - t0
    output_text = proc.stdout or ""

    # Re-create log parent because the final output directory may have been
    # touched by helper processes on Windows.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"Script: {script}\n"
        f"Return code: {proc.returncode}\n"
        f"Elapsed seconds: {elapsed:.6f}\n\n"
        f"{output_text}",
        encoding="utf-8",
    )

    print(output_text)
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Log: {log_path}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed with return code {proc.returncode}. "
            f"See {log_path}"
        )

def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing expected result: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def save_csv_tex(
    df: pd.DataFrame,
    stem: str,
    table_dir: Path,
    *,
    caption: str | None = None,
    label: str | None = None,
    index: bool = False,
    float_format: str = "%.4f",
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"{stem}.csv"
    tex_path = table_dir / f"{stem}.tex"

    df.to_csv(csv_path, index=index)

    latex = df.to_latex(
        index=index,
        escape=True,
        float_format=lambda x: float_format % x,
        caption=caption,
        label=label,
    )
    tex_path.write_text(latex, encoding="utf-8")


def save_figure(fig, stem: str, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_raw_csvs(source_dir: Path, dest_dir: Path, prefix: str) -> None:
    """Copy raw CSV outputs defensively across platforms.

    The destination is recreated here (rather than only once at startup)
    because external experiment engines may create/remove output folders and
    Windows CopyFile2 raises WinError 3 if the destination parent disappeared.
    """
    if not source_dir.exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(source_dir.glob("*.csv")):
        if not p.is_file():
            continue

        dest = dest_dir / f"{prefix}_{p.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(p, dest)
        except FileNotFoundError:
            # Re-check both sides so the error message is useful on Windows.
            if not p.exists():
                raise FileNotFoundError(f"Raw source disappeared before copy: {p}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)


def pct_reduction(proposed: float, baseline: float) -> float:
    if not np.isfinite(proposed) or not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return 100.0 * (baseline - proposed) / baseline


def normalize_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].astype(str).str.lower()
    return out


def nice_dataset(x: str) -> str:
    return str(x).strip().capitalize()


def nice_protocol(x: str) -> str:
    mapping = {
        "stratified_row_cv": "Sample-level CV",
        "strict_rp_grouped_cv": "RP-grouped CV",
        "strict_grouped_split": "Strict split conformal",
        "strict_grouped_split_rp_corrected": "Strict split conformal (RP-corrected)",
        "full_train_grouped_oof_empirical": "Grouped-OOF empirical",
    }
    return mapping.get(str(x), str(x).replace("_", " "))


# =============================================================================
# LOAD VALIDATED INTERMEDIATE RESULTS
# =============================================================================

def load_original_results(project_root: Path) -> dict[str, pd.DataFrame]:
    table_dir = project_root / "paper_results_all_contributions" / "tables"

    names = {
        "dataset_summary": "dataset_summary.csv",
        "point": "point_localization_results.csv",
        "published": "published_comparison.csv",
        "cv_folds": "grouped_cv_fold_results.csv",
        "cv_summary": "grouped_cv_summary.csv",
        "split_uncertainty": "split_conformal_results.csv",
        "oof_uncertainty": "full_oof_uncertainty_results.csv",
        "bootstrap": "cluster_bootstrap_comparison.csv",
        "data_utilization": "data_utilization_ablation.csv",
        "contribution": "contribution_summary.csv",
    }

    return {
        key: normalize_dataset_names(read_csv(table_dir / filename))
        for key, filename in names.items()
    }


def load_eetac_results(project_root: Path) -> dict[str, pd.DataFrame]:
    out_dir = project_root / "eetac_external_validation_results"

    names = {
        "dataset_summary": "dataset_summary.csv",
        "audit": "ap_bssid_audit.csv",
        "scenario": "scenario_definition.csv",
        "folds": "fold_results.csv",
        "summary": "summary_results.csv",
        "improvement": "paired_improvement_summary.csv",
        "cross_day": "cross_day_results.csv",
        "cross_improvement": "cross_day_improvement_summary.csv",
    }

    return {
        key: read_csv(out_dir / filename)
        for key, filename in names.items()
    }


# =============================================================================
# TABLE 1: DATASET SUMMARY
# =============================================================================

def build_table01(
    original: dict[str, pd.DataFrame],
    eetac: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    ds = original["dataset_summary"].copy()

    rows = []
    for dataset in ORIGINAL_DATASETS:
        g = ds[ds["dataset"] == dataset]
        if g.empty:
            continue
        r = g.iloc[0]

        rows.append({
            "Dataset": nice_dataset(dataset),
            "Environment": SOURCE_AREA[dataset],
            "Grid spacing (m)": SOURCE_GRID_M[dataset],
            "Train samples": int(r["train_rows"]),
            "Test samples": int(r["test_rows"]),
            "Train RPs": int(r["train_reference_points"]),
            "Test RPs": int(r["test_reference_points"]),
            "APs": int(r["n_aps"]),
            "Features": int(r["n_features"]),
        })

    eds = eetac["dataset_summary"].copy()
    for _, r in eds.sort_values("day").iterrows():
        day = int(r["day"])
        rows.append({
            "Dataset": f"EETAC Day {day}",
            "Environment": "19.2 x 8 m^2 auditorium",
            "Grid spacing (m)": np.nan,
            "Train samples": int(r["rows"]),
            "Test samples": np.nan,
            "Train RPs": int(r["reference_points"]),
            "Test RPs": np.nan,
            "APs": np.nan,
            "Features": np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 2: MODEL CONFIGURATION
# =============================================================================

def build_table02() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Configuration": "RF300-F1.0",
            "Role": "Full-feature baseline",
            "Trees": N_TREES,
            "max_features": BASELINE_MAX_FEATURES,
            "Maximum depth": "None",
            "Minimum leaf": 1,
            "Bootstrap": "Yes",
            "RP-grouped selection": "No retuning",
        },
        {
            "Configuration": "RF300-F0.7",
            "Role": "Proposed random-subspace RF",
            "Trees": N_TREES,
            "max_features": PROPOSED_MAX_FEATURES,
            "Maximum depth": "None",
            "Minimum leaf": 1,
            "Bootstrap": "Yes",
            "RP-grouped selection": "Frozen before EETAC",
        },
    ])


# =============================================================================
# TABLE 3: ORIGINAL-DATASET ACCURACY
# =============================================================================

def build_table03(original: dict[str, pd.DataFrame]) -> pd.DataFrame:
    point = original["point"].copy()
    published = original["published"].copy()
    bootstrap = original["bootstrap"].copy()

    rows = []

    for dataset in ORIGINAL_DATASETS:
        pg = point[point["dataset"] == dataset]
        pub = published[published["dataset"] == dataset]
        bs = bootstrap[bootstrap["dataset"] == dataset]

        if pg.empty:
            continue

        b = pg[pg["model"] == "RF300-F1.0"].iloc[0]
        p = pg[pg["model"] == "RF300-F0.7"].iloc[0]

        published_value = (
            float(pub.iloc[0]["published_hybrid_metric_m"])
            if not pub.empty
            else PUBLISHED_ORIGINAL[dataset]
        )

        rows.append({
            "Dataset": nice_dataset(dataset),
            "Published metric (m)": published_value,
            "Reproduced F1.0 (m)": float(b["paper_metric_m"]),
            "Proposed F0.7 (m)": float(p["paper_metric_m"]),
            "Improvement vs F1.0 (%)": pct_reduction(
                float(p["paper_metric_m"]),
                float(b["paper_metric_m"]),
            ),
            "2D RMSE F1.0 (m)": float(b["rmse_euclidean_m"]),
            "2D RMSE F0.7 (m)": float(p["rmse_euclidean_m"]),
            "P90 F1.0 (m)": float(b["p90_euclidean_m"]),
            "P90 F0.7 (m)": float(p["p90_euclidean_m"]),
            # Paired cluster bootstrap over test RPs (F0.7 - F1.0, metres).
            "Diff 95% CI low (m)": float(bs.iloc[0]["ci95_low_m"]) if not bs.empty else np.nan,
            "Diff 95% CI high (m)": float(bs.iloc[0]["ci95_high_m"]) if not bs.empty else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 4: RP-GROUPED CV
# =============================================================================

def build_table04(original: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cv = original["cv_summary"].copy()
    rows = []

    for dataset in ORIGINAL_DATASETS:
        g = cv[cv["dataset"] == dataset]
        if g.empty:
            continue

        b = g[g["model"] == "RF300-F1.0"].iloc[0]
        p = g[g["model"] == "RF300-F0.7"].iloc[0]

        rows.append({
            "Dataset": nice_dataset(dataset),
            "F1.0 paper metric (m)": float(b["paper_metric_m"]),
            "F0.7 paper metric (m)": float(p["paper_metric_m"]),
            "Improvement (%)": pct_reduction(
                float(p["paper_metric_m"]),
                float(b["paper_metric_m"]),
            ),
            "F1.0 2D RMSE (m)": float(b["rmse_euclidean_m"]),
            "F0.7 2D RMSE (m)": float(p["rmse_euclidean_m"]),
            "F1.0 P90 (m)": float(b["p90_euclidean_m"]),
            "F0.7 P90 (m)": float(p["p90_euclidean_m"]),
            "F1.0 P95 (m)": float(b["p95_euclidean_m"]),
            "F0.7 P95 (m)": float(p["p95_euclidean_m"]),
        })

    return pd.DataFrame(rows)


# =============================================================================
# EETAC MAIN-SCENARIO UTILITIES
# =============================================================================

def available_main_eetac_scenarios(
    eetac: dict[str, pd.DataFrame],
) -> list[str]:
    available = set(eetac["summary"]["scenario"].astype(str))
    preferred = [
        s for s in EETAC_MAIN_SCENARIOS
        if s in available
    ]

    # If High was named differently in an older validated output, retain only
    # a clearly 7-AP scenario.  Do not auto-select based on accuracy.
    if "High_4Google_3Linksys" not in preferred:
        scenario_df = eetac["scenario"].copy()
        if not scenario_df.empty:
            high = scenario_df[
                (scenario_df["n_physical_aps"] == 7)
                &
                (scenario_df["scenario"].astype(str).str.contains("High"))
            ]
            if len(high) == 1:
                candidate = str(high.iloc[0]["scenario"])
                if candidate in available:
                    preferred.append(candidate)

    return list(dict.fromkeys(preferred))


# =============================================================================
# TABLE 5: EETAC EXTERNAL VALIDATION
# =============================================================================

def build_table05(eetac: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = eetac["summary"].copy()
    scenarios = available_main_eetac_scenarios(eetac)

    rows = []

    for scenario in scenarios:
        for protocol in ("stratified_row_cv", "strict_rp_grouped_cv"):
            g = summary[
                (summary["scenario"] == scenario)
                &
                (summary["protocol"] == protocol)
            ]

            if g.empty:
                continue

            b = g[g["model"] == "RF300-F1.0"].iloc[0]
            p = g[g["model"] == "RF300-F0.7"].iloc[0]

            rows.append({
                "Scenario": scenario,
                "Protocol": nice_protocol(protocol),
                "F1.0 2D RMSE (m)": float(b["rmse_2d_m"]),
                "F0.7 2D RMSE (m)": float(p["rmse_2d_m"]),
                "RMSE improvement (%)": pct_reduction(
                    float(p["rmse_2d_m"]),
                    float(b["rmse_2d_m"]),
                ),
                "F1.0 P90 (m)": float(b["p90_error_m"]),
                "F0.7 P90 (m)": float(p["p90_error_m"]),
                "F1.0 P95 (m)": float(b["p95_error_m"]),
                "F0.7 P95 (m)": float(p["p95_error_m"]),
            })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 6: CROSS-DAY
# =============================================================================

def build_table06(eetac: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cd = eetac["cross_improvement"].copy()
    scenarios = available_main_eetac_scenarios(eetac)

    rows = []

    for scenario in scenarios:
        g = cd[cd["scenario"] == scenario]
        for _, r in g.sort_values(["train_day", "test_day"]).iterrows():
            rows.append({
                "Scenario": scenario,
                "Direction": f"Day {int(r['train_day'])} -> Day {int(r['test_day'])}",
                "F1.0 2D RMSE (m)": float(r["baseline_rmse_2d_m"]),
                "F0.7 2D RMSE (m)": float(r["proposed_rmse_2d_m"]),
                "RMSE improvement (%)": float(r["rmse_reduction_pct"]),
                "F1.0 P90 (m)": float(r["baseline_p90_m"]),
                "F0.7 P90 (m)": float(r["proposed_p90_m"]),
                "P90 improvement (%)": float(r["p90_reduction_pct"]),
            })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 7: EFFICIENCY
# =============================================================================

def build_table07(
    original: dict[str, pd.DataFrame],
    eetac: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    # Original official-test timing.
    point = original["point"].copy()
    for dataset in ORIGINAL_DATASETS:
        g = point[point["dataset"] == dataset]
        if g.empty:
            continue
        for _, r in g.sort_values("model").iterrows():
            rows.append({
                "Dataset / scenario": nice_dataset(dataset),
                "Protocol": "Official test",
                "Model": r["model"],
                "Accuracy metric": float(r["paper_metric_m"]),
                "Accuracy metric type": "Paper-compatible error (m)",
                "Train time (s)": float(r["train_time_s"]),
                "Prediction time (s)": float(r["predict_time_s"]),
                "Inference (us/sample)": float(r["predict_us_per_sample"]),
                "Model size (MB)": float(r["model_size_mb"]),
            })

    # EETAC main scenarios: average CV timing, model size unavailable.
    summary = eetac["summary"].copy()
    scenarios = available_main_eetac_scenarios(eetac)
    for scenario in scenarios:
        g = summary[
            (summary["scenario"] == scenario)
            &
            (summary["protocol"] == "stratified_row_cv")
        ]
        for _, r in g.sort_values("model").iterrows():
            rows.append({
                "Dataset / scenario": f"EETAC {scenario}",
                "Protocol": "Sample-level CV",
                "Model": r["model"],
                "Accuracy metric": float(r["rmse_2d_m"]),
                "Accuracy metric type": "2D RMSE (m)",
                "Train time (s)": float(r["train_time_s"]),
                "Prediction time (s)": float(r["predict_time_s"]),
                "Inference (us/sample)": np.nan,
                "Model size (MB)": np.nan,
            })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 8: DATA UTILIZATION
# =============================================================================

def build_table08(original: dict[str, pd.DataFrame]) -> pd.DataFrame:
    du = original["data_utilization"].copy()
    rows = []

    for dataset in ORIGINAL_DATASETS:
        g = du[du["dataset"] == dataset]
        if g.empty:
            continue

        fit = g[g["training_scope"] == "grouped_fit_subset"].iloc[0]
        full = g[g["training_scope"] == "100pct_official_train"].iloc[0]

        rows.append({
            "Dataset": nice_dataset(dataset),
            "Fit-subset RPs": int(fit["train_rps"]),
            "100% train RPs": int(full["train_rps"]),
            "Fit-subset metric (m)": float(fit["paper_metric_m"]),
            "100% metric (m)": float(full["paper_metric_m"]),
            "Improvement (%)": pct_reduction(
                float(full["paper_metric_m"]),
                float(fit["paper_metric_m"]),
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 9: UNCERTAINTY
# =============================================================================

def build_table09(original: dict[str, pd.DataFrame]) -> pd.DataFrame:
    strict = original["split_uncertainty"].copy()
    oof = original["oof_uncertainty"].copy()

    rows = []

    # Main paper comparison: proposed circle vs proposed ellipse.
    for dataset in ORIGINAL_DATASETS:
        for confidence in CONFIDENCE_LEVELS:
            # Strict.
            g = strict[
                (strict["dataset"] == dataset)
                &
                (np.isclose(strict["confidence"], confidence))
                &
                (strict["predictor"] == "RF300-F0.7")
                &
                (strict["region"].isin(["circle", "ellipse"]))
            ]

            for _, r in g.sort_values(["protocol", "region"]).iterrows():
                rows.append({
                    "Dataset": nice_dataset(dataset),
                    "Protocol": nice_protocol(r["protocol"]),
                    "Confidence": confidence,
                    "Region": str(r["region"]).capitalize(),
                    "Coverage": float(r["empirical_coverage"]),
                    "Area (m^2)": float(r["area_m2"]),
                    "Calibration RPs": int(r["calibration_rps"]),
                })

            # Full-training OOF empirical.
            g2 = oof[
                (oof["dataset"] == dataset)
                &
                (np.isclose(oof["confidence"], confidence))
                &
                (oof["region"].isin(["circle", "ellipse"]))
            ]

            for _, r in g2.sort_values("region").iterrows():
                rows.append({
                    "Dataset": nice_dataset(dataset),
                    "Protocol": "Grouped-OOF empirical",
                    "Confidence": confidence,
                    "Region": str(r["region"]).capitalize(),
                    "Coverage": float(r["empirical_coverage"]),
                    "Area (m^2)": float(r["area_m2"]),
                })

    return pd.DataFrame(rows)


# =============================================================================
# FIGURE 6: ORIGINAL ACCURACY
# =============================================================================

def fig06_original_accuracy(table03: pd.DataFrame, figure_dir: Path) -> None:
    df = table03.copy()
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        x - width,
        df["Published metric (m)"],
        width,
        label="Published hybrid RF",
    )
    ax.bar(
        x,
        df["Reproduced F1.0 (m)"],
        width,
        label="Reproduced RF300-F1.0",
    )
    ax.bar(
        x + width,
        df["Proposed F0.7 (m)"],
        width,
        label="Proposed RF300-F0.7",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"])
    ax.set_ylabel("Paper-compatible localization error (m)")
    ax.set_title("Original benchmark: reproduction and proposed model")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "fig06_original_accuracy", figure_dir)


# =============================================================================
# FIGURE 7: COMPOSITE ORIGINAL CDF FIGURE
# =============================================================================

def compose_existing_pngs(
    pngs: list[Path],
    labels: list[str],
    stem: str,
    figure_dir: Path,
) -> None:
    existing = [(p, lab) for p, lab in zip(pngs, labels) if p.exists()]
    if not existing:
        return

    fig, axes = plt.subplots(
        1,
        len(existing),
        figsize=(5.2 * len(existing), 4.2),
        squeeze=False,
    )

    for ax, (p, label) in zip(axes[0], existing):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label)

    save_figure(fig, stem, figure_dir)


def fig07_error_cdf(project_root: Path, figure_dir: Path) -> None:
    plot_dir = project_root / "paper_results_all_contributions" / "plots"
    pngs = [
        plot_dir / f"{dataset}_result_error_cdf.png"
        for dataset in ORIGINAL_DATASETS
    ]
    labels = [nice_dataset(d) for d in ORIGINAL_DATASETS]
    compose_existing_pngs(
        pngs,
        labels,
        "fig07_error_cdf",
        figure_dir,
    )


# =============================================================================
# FIGURE 8: COMPOSITE GROUPED-CV FIGURE
# =============================================================================

def fig08_grouped_cv(project_root: Path, figure_dir: Path) -> None:
    plot_dir = project_root / "paper_results_all_contributions" / "plots"
    pngs = [
        plot_dir / f"{dataset}_result_grouped_cv_folds.png"
        for dataset in ORIGINAL_DATASETS
    ]
    labels = [nice_dataset(d) for d in ORIGINAL_DATASETS]
    compose_existing_pngs(
        pngs,
        labels,
        "fig08_grouped_cv",
        figure_dir,
    )


# =============================================================================
# FIGURE 9: EETAC EXTERNAL VALIDATION
# =============================================================================

def fig09_eetac(table05: pd.DataFrame, figure_dir: Path) -> None:
    if table05.empty:
        return

    # Use one category per scenario/protocol.
    labels = (
        table05["Scenario"].astype(str)
        + "\n"
        + table05["Protocol"].astype(str)
    )
    x = np.arange(len(table05))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(9, len(table05) * 2.0), 5.4))
    ax.bar(
        x - width / 2,
        table05["F1.0 2D RMSE (m)"],
        width,
        label="RF300-F1.0",
    )
    ax.bar(
        x + width / 2,
        table05["F0.7 2D RMSE (m)"],
        width,
        label="RF300-F0.7",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("2D RMSE (m)")
    ax.set_title("Independent EETAC external validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "fig09_eetac_external_validation", figure_dir)


# =============================================================================
# FIGURE 10: ACCURACY-EFFICIENCY
# =============================================================================

def fig10_accuracy_efficiency(
    original: dict[str, pd.DataFrame],
    figure_dir: Path,
) -> None:
    point = original["point"].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    for model in ("RF300-F1.0", "RF300-F0.7"):
        g = point[point["model"] == model]
        ax.scatter(
            g["predict_us_per_sample"],
            g["paper_metric_m"],
            s=80,
            label=model,
        )

        for _, r in g.iterrows():
            ax.annotate(
                nice_dataset(r["dataset"]),
                (
                    float(r["predict_us_per_sample"]),
                    float(r["paper_metric_m"]),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Inference time (us/sample)")
    ax.set_ylabel("Paper-compatible localization error (m)")
    ax.set_title("Accuracy-efficiency trade-off on the original benchmark")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, "fig10_accuracy_efficiency", figure_dir)


# =============================================================================
# FIGURE 11: CROSS-DAY
# =============================================================================

def fig11_cross_day(table06: pd.DataFrame, figure_dir: Path) -> None:
    if table06.empty:
        return

    labels = (
        table06["Scenario"].astype(str)
        + "\n"
        + table06["Direction"].astype(str)
    )
    x = np.arange(len(table06))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(9, 2.0 * len(table06)), 5.4))
    ax.bar(
        x - width / 2,
        table06["F1.0 2D RMSE (m)"],
        width,
        label="RF300-F1.0",
    )
    ax.bar(
        x + width / 2,
        table06["F0.7 2D RMSE (m)"],
        width,
        label="RF300-F0.7",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("2D RMSE (m)")
    ax.set_title("Cross-day temporal generalization")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "fig11_cross_day", figure_dir)


# =============================================================================
# FIGURE 12: UNCERTAINTY AREA-COVERAGE
# =============================================================================

def fig12_uncertainty(table09: pd.DataFrame, figure_dir: Path) -> None:
    if table09.empty:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    marker_map = {
        "Circle": "o",
        "Ellipse": "s",
    }

    for (protocol, region), g in table09.groupby(["Protocol", "Region"]):
        ax.scatter(
            g["Area (m^2)"],
            g["Coverage"] * 100.0,
            s=65,
            marker=marker_map.get(region, "o"),
            label=f"{protocol} - {region}",
        )

        for _, r in g.iterrows():
            ax.annotate(
                f"{r['Dataset']} {int(round(float(r['Confidence']) * 100))}%",
                (
                    float(r["Area (m^2)"]),
                    float(r["Coverage"]) * 100.0,
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    ax.axhline(90, linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(95, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Prediction-region area (m^2)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Uncertainty-region area versus empirical coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    save_figure(
        fig,
        "fig12_uncertainty_area_coverage",
        figure_dir,
    )


# =============================================================================
# METADATA / AUDIT
# =============================================================================

def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:
        return f"unknown ({exc})"


def write_protocol_metadata(
    project_root: Path,
    paper_results: Path,
    original_helper: Path,
    eetac_helper: Path,
) -> None:
    paper_results.mkdir(parents=True, exist_ok=True)

    environment = {
        "run_timestamp_local": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "scikit-learn": package_version("scikit-learn"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "joblib": package_version("joblib"),
        "pandas": package_version("pandas"),
        "matplotlib": package_version("matplotlib"),
    }

    meta = {
        "seed": SEED,
        "n_trees": N_TREES,
        "baseline_max_features": BASELINE_MAX_FEATURES,
        "proposed_max_features": PROPOSED_MAX_FEATURES,
        "grouped_cv_folds": CV_FOLDS,
        "calibration_fraction": CALIBRATION_FRACTION,
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "project_root": str(project_root),
        "original_engine": str(original_helper),
        "eetac_engine": str(eetac_helper),
        "main_eetac_scenarios_requested": list(EETAC_MAIN_SCENARIOS),
        "environment": environment,
        "scientific_notes": [
            "F=0.7 is frozen before EETAC evaluation.",
            "Original published comparison uses paper-compatible metric.",
            "EETAC uses true 2D RMSE in metres.",
            "Sample-level and RP-grouped EETAC protocols are separate.",
            "Grouped-OOF empirical uncertainty is not ordinary split conformal.",
            "EETAC Low/2-AP scenario is excluded from main tables.",
        ],
    }

    (paper_results / "protocol_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    env_lines = [
        "FINAL PAPER EXPERIMENT ENVIRONMENT",
        "=" * 40,
        *[f"{k}: {v}" for k, v in environment.items()],
    ]
    (paper_results / "environment.txt").write_text(
        "\n".join(env_lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)

    ensure_dataset_files(project_root)

    original_helper = find_helper(
        project_root,
        "paper_all_contributions_wifi_rtt_rss.py",
        "original_dataset",
    )
    eetac_helper = find_helper(
        project_root,
        "eetac_external_validation_subspace_rf_v2.py",
        "eetac",
    )

    paper_results = project_root / "paper_results"

    if paper_results.exists() and not args.keep_paper_results:
        try:
            shutil.rmtree(paper_results)
        except PermissionError:
            # On Windows a viewer/editor may temporarily hold a generated
            # file open.  Preserve the folder and overwrite outputs below.
            print("[warning] Could not fully remove paper_results; "
                  "existing files will be overwritten where possible.")

    table_dir = paper_results / "tables"
    figure_dir = paper_results / "figures"
    raw_dir = paper_results / "raw"
    log_dir = paper_results / "logs"

    for p in (table_dir, figure_dir, raw_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FINAL PAPER EXPERIMENT SUITE")
    print("=" * 100)
    print(f"Project root : {project_root}")
    print(f"Dataset dir  : {project_root / 'dataset'}")
    print(f"Baseline     : RF300-F1.0")
    print(f"Proposed     : RF300-F0.7")
    print(f"Seed         : {SEED}")
    print(f"Build only   : {args.build_only}")

    suite_t0 = time.perf_counter()

    if not args.build_only:
        run_engine(
            original_helper,
            project_root,
            log_dir / "01_original_dataset_engine.log",
        )
        run_engine(
            eetac_helper,
            project_root,
            log_dir / "02_eetac_engine.log",
        )

    # Verify intermediate outputs exist.
    original_intermediate = project_root / "paper_results_all_contributions"
    eetac_intermediate = project_root / "eetac_external_validation_results"

    if not original_intermediate.exists():
        raise FileNotFoundError(
            "Missing original intermediate results directory: "
            f"{original_intermediate}"
        )
    if not eetac_intermediate.exists():
        raise FileNotFoundError(
            "Missing EETAC intermediate results directory: "
            f"{eetac_intermediate}"
        )

    original = load_original_results(project_root)
    eetac = load_eetac_results(project_root)

    # Recreate final output directories after the external engines finish.
    # This is intentionally repeated for robust Windows behavior.
    for p in (table_dir, figure_dir, raw_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Preserve raw numerical outputs for audit/reproducibility.
    copy_raw_csvs(
        original_intermediate / "tables",
        raw_dir,
        "original",
    )
    copy_raw_csvs(
        eetac_intermediate,
        raw_dir,
        "eetac",
    )

    # -------------------------------------------------------------------------
    # TABLES 1-9
    # -------------------------------------------------------------------------
    table01 = build_table01(original, eetac)
    table02 = build_table02()
    table03 = build_table03(original)
    table04 = build_table04(original)
    table05 = build_table05(eetac)
    table06 = build_table06(eetac)
    table07 = build_table07(original, eetac)
    table08 = build_table08(original)
    table09 = build_table09(original)

    save_csv_tex(
        table01,
        "table01_dataset_summary",
        table_dir,
        caption="Datasets used in the final experiments.",
        label="tab:final_dataset_summary",
    )
    save_csv_tex(
        table02,
        "table02_model_configuration",
        table_dir,
        caption="Random Forest configurations used in the primary comparison.",
        label="tab:final_model_configuration",
    )
    save_csv_tex(
        table03,
        "table03_original_accuracy",
        table_dir,
        caption=(
            "Point-localization performance on the original hybrid "
            "RTT--RSS benchmark."
        ),
        label="tab:original_accuracy",
    )
    save_csv_tex(
        table04,
        "table04_grouped_cv",
        table_dir,
        caption="Five-fold RP-grouped validation on the original benchmark.",
        label="tab:grouped_cv",
    )
    save_csv_tex(
        table05,
        "table05_eetac_external_validation",
        table_dir,
        caption=(
            "External validation on the independent EETAC auditorium dataset."
        ),
        label="tab:eetac_external",
    )
    save_csv_tex(
        table06,
        "table06_cross_day",
        table_dir,
        caption="Cross-day EETAC temporal-generalization results.",
        label="tab:cross_day",
    )
    save_csv_tex(
        table07,
        "table07_efficiency",
        table_dir,
        caption="Accuracy and computational-efficiency measurements.",
        label="tab:efficiency",
    )
    save_csv_tex(
        table08,
        "table08_data_utilization",
        table_dir,
        caption=(
            "Effect of training the proposed predictor on the grouped fitting "
            "subset versus all official training reference points."
        ),
        label="tab:data_utilization",
    )
    save_csv_tex(
        table09,
        "table09_uncertainty",
        table_dir,
        caption=(
            "Circular and covariance-aware elliptical uncertainty-region "
            "performance."
        ),
        label="tab:uncertainty",
    )

    # -------------------------------------------------------------------------
    # FIGURES 6-12
    # -------------------------------------------------------------------------
    fig06_original_accuracy(table03, figure_dir)
    fig07_error_cdf(project_root, figure_dir)
    fig08_grouped_cv(project_root, figure_dir)
    fig09_eetac(table05, figure_dir)
    fig10_accuracy_efficiency(original, figure_dir)
    fig11_cross_day(table06, figure_dir)
    fig12_uncertainty(table09, figure_dir)

    write_protocol_metadata(
        project_root,
        paper_results,
        original_helper,
        eetac_helper,
    )

    expected_table_stems = [
        "table01_dataset_summary",
        "table02_model_configuration",
        "table03_original_accuracy",
        "table04_grouped_cv",
        "table05_eetac_external_validation",
        "table06_cross_day",
        "table07_efficiency",
        "table08_data_utilization",
        "table09_uncertainty",
    ]
    expected_figure_stems = [
        "fig06_original_accuracy",
        "fig07_error_cdf",
        "fig08_grouped_cv",
        "fig09_eetac_external_validation",
        "fig10_accuracy_efficiency",
        "fig11_cross_day",
        "fig12_uncertainty_area_coverage",
    ]

    missing_outputs = []
    for stem in expected_table_stems:
        for suffix in (".csv", ".tex"):
            p = table_dir / f"{stem}{suffix}"
            if not p.exists():
                missing_outputs.append(str(p))

    for stem in expected_figure_stems:
        for suffix in (".pdf", ".png"):
            p = figure_dir / f"{stem}{suffix}"
            if not p.exists():
                missing_outputs.append(str(p))

    if missing_outputs:
        print("\n[warning] The run completed but some predefined manuscript outputs "
              "were not created:")
        for p in missing_outputs:
            print(f"  - {p}")
    else:
        print("\nVerified: all 9 paper tables and all 7 paper figures were created "
              "in both requested formats.")

    elapsed = time.perf_counter() - suite_t0

    final_log = [
        "FINAL PAPER EXPERIMENT SUITE COMPLETED",
        f"Elapsed seconds: {elapsed:.6f}",
        f"Project root: {project_root}",
        f"Paper results: {paper_results}",
        "",
        "Main EETAC scenarios actually included:",
        *[
            f"  - {s}"
            for s in available_main_eetac_scenarios(eetac)
        ],
        "",
        "Generated tables:",
        *[
            f"  - {p.name}"
            for p in sorted(table_dir.glob("*.csv"))
        ],
        "",
        "Generated figures:",
        *[
            f"  - {p.name}"
            for p in sorted(figure_dir.glob("*.pdf"))
        ],
    ]

    (log_dir / "00_final_suite_summary.log").write_text(
        "\n".join(final_log),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Elapsed      : {elapsed:.2f} s")
    print(f"Tables       : {table_dir}")
    print(f"Figures      : {figure_dir}")
    print(f"Raw outputs  : {raw_dir}")
    print(f"Logs         : {log_dir}")
    print(f"Environment  : {paper_results / 'environment.txt'}")
    print(f"Metadata     : {paper_results / 'protocol_metadata.json'}")
    print("\nNext step: run scripts/extensions/make_paper_tables.py to rebuild the LaTeX tables (see README).")


if __name__ == "__main__":
    main()
