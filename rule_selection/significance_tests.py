"""
Mann-Whitney U significance tests over the 3 causal deltas (Delta_generador,
Delta_politica_TopSupport, Delta_politica_GENESIS) per log, on the per-replica PCE
values behind each arm's `final_comparison.csv` row (CLAUDE.md Sec. 28/29).

Per-replica PCE is not stored anywhere small -- it only exists inside the raw
`data/4.simulation_results/<log>/<run_label>_rep*/*_prosimos_stats.csv` files
(hundreds of MB across all replicas/arms/logs, intentionally not tracked in git).
This script extracts just the one PCE float per replica (processing_time /
cycle_time * 100, same formula as `kpi.mean_pce_over_replicas`) and writes that
small, reproducible summary to `data/5.rule_selection/<log>/per_replica_pce.csv`,
then runs the significance tests on it and (re)writes
`data/5.rule_selection/_consolidated/significance_tests_mannwhitney.csv`.

Bonferroni correction is applied within each log (3 comparisons per log, not 9
across all logs) -- each log is treated as an independent experiment (CLAUDE.md
Sec. 28). Effect size is the rank-biserial correlation.

Usage:
    python -m rule_selection.significance_tests
    python -m rule_selection.significance_tests --log RunningExample
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

from rule_selection import artifacts
from rule_selection.kpi import analizar_metricas_archivo

ROOT = Path(__file__).resolve().parent.parent
SIMULATION_RESULTS_DIR = ROOT / "data" / "4.simulation_results"
CONSOLIDATED_DIR = ROOT / "data" / "5.rule_selection" / "_consolidated"

# Arm -> run_label. GENESIS has no fixed label: it reuses whichever candidate won
# that log's screening (`screening_<candidate_id>`), read from screening_summary.csv.
FIXED_ARM_LABELS = {
    "AS-IS": "final_asis",
    "Placebo": "final_placebo",
    "Top-Support": "final_topsupport",
}


def _genesis_run_label(log_name: str) -> str:
    summary_path = artifacts.rule_selection_dir(ROOT, log_name) / "screening_summary.csv"
    df = pd.read_csv(summary_path)
    winner = df.iloc[0]["candidate_id"]
    return f"screening_{winner}"


def _arm_run_labels(log_name: str) -> dict:
    labels = dict(FIXED_ARM_LABELS)
    labels["GENESIS"] = _genesis_run_label(log_name)
    return labels


def _pce_from_stats_file(stats_csv_path: Path) -> float:
    df = analizar_metricas_archivo(stats_csv_path)
    processing_time = df.loc["processing_time", "Average"]
    cycle_time = df.loc["cycle_time", "Average"]
    return float(processing_time / cycle_time * 100)


def collect_per_replica_pce(log_name: str) -> pd.DataFrame:
    """Extracts one PCE float per replica for each of the 4 arms of `log_name`."""
    rows = []
    for arm, run_label in _arm_run_labels(log_name).items():
        rep_dirs = sorted(
            SIMULATION_RESULTS_DIR.joinpath(log_name).glob(f"{run_label}_rep*")
        )
        if not rep_dirs:
            raise FileNotFoundError(
                f"No replica folders found for arm={arm!r} run_label={run_label!r} "
                f"under {SIMULATION_RESULTS_DIR / log_name}"
            )
        for rep_dir in rep_dirs:
            stats_files = list(rep_dir.glob("*_prosimos_stats.csv"))
            if not stats_files:
                raise FileNotFoundError(f"No *_prosimos_stats.csv in {rep_dir}")
            replica = rep_dir.name.rsplit("_rep", 1)[-1]
            rows.append(
                {
                    "log": log_name,
                    "arm": arm,
                    "run_label": run_label,
                    "replica": int(replica),
                    "pce": _pce_from_stats_file(stats_files[0]),
                }
            )
    return pd.DataFrame(rows)


def _rank_biserial(x: list, y: list, u_stat: float) -> float:
    """r = 1 - 2U / (n_x * n_y), the common effect size for Mann-Whitney U."""
    return 1.0 - (2.0 * u_stat) / (len(x) * len(y))


def compute_significance(per_replica: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("Delta_generador", "Placebo", "AS-IS"),
        ("Delta_politica_TopSupport", "Top-Support", "Placebo"),
        ("Delta_politica_GENESIS", "GENESIS", "Placebo"),
    ]

    rows = []
    for log_name, group in per_replica.groupby("log"):
        pce_by_arm = {arm: g.sort_values("replica")["pce"].tolist() for arm, g in group.groupby("arm")}
        raw_results = []
        for delta_name, arm_x, arm_y in comparisons:
            x, y = pce_by_arm[arm_x], pce_by_arm[arm_y]
            u_stat, p_raw = mannwhitneyu(x, y, alternative="two-sided")
            raw_results.append(
                {
                    "Log": log_name,
                    "Delta": delta_name,
                    "n_x": len(x),
                    "n_y": len(y),
                    "mean_x": sum(x) / len(x),
                    "mean_y": sum(y) / len(y),
                    "U": u_stat,
                    "p_raw": p_raw,
                    "rank_biserial_r": _rank_biserial(x, y, u_stat),
                }
            )
        n_comparisons = len(raw_results)
        for r in raw_results:
            r["p_bonferroni_per_log"] = min(r["p_raw"] * n_comparisons, 1.0)
            r["significant_0.05_bonf"] = r["p_bonferroni_per_log"] < 0.05
        rows.extend(raw_results)

    columns = [
        "Log", "Delta", "n_x", "n_y", "mean_x", "mean_y", "U", "p_raw",
        "p_bonferroni_per_log", "significant_0.05_bonf", "rank_biserial_r",
    ]
    return pd.DataFrame(rows)[columns]


def main(log_names: list[str] | None = None) -> None:
    log_names = log_names or ["RunningExample", "PurchasingExample", "BPI_Challenge_2012"]

    per_replica_frames = []
    for log_name in log_names:
        df = collect_per_replica_pce(log_name)
        per_replica_frames.append(df)
        out_path = artifacts.rule_selection_dir(ROOT, log_name) / "per_replica_pce.csv"
        df.to_csv(out_path, index=False)
        print(f"[{log_name}] wrote {len(df)} per-replica PCE rows -> {out_path}")

    per_replica = pd.concat(per_replica_frames, ignore_index=True)
    significance = compute_significance(per_replica)

    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONSOLIDATED_DIR / "significance_tests_mannwhitney.csv"
    significance.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {len(significance)} rows -> {out_path}")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", dest="logs", default=None,
                         help="Restrict to one log (repeatable). Default: all 3.")
    args = parser.parse_args()
    main(args.logs)
