"""
Convencion de rutas de salida del pipeline de seleccion de reglas: carpeta nueva
`data/5.rule_selection/<log_name>/`, consistente con la numeracion existente
(`data/0.logs`, ..., `data/4.simulation_results`).

Los `.ini` de cada candidata siguen viviendo en `data/0.logs/<log_name>/` (via
`Params.routes["rules"]`, solo cambiando `rules_filename` por candidata -- ver
`simulation_arms.make_candidate_params`), y las salidas de Simod/Prosimos siguen en
las carpetas existentes (`data/3.bps_tobe`, `data/4.simulation_results`, etc.) -- no
se inventa una convencion paralela para esas dos, solo para los CSV de reporte de
este pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def rule_selection_dir(root: Path, log_name: str) -> Path:
    path = Path(root) / "data" / "5.rule_selection" / log_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_candidates(
    df: pd.DataFrame, root: Path, log_name: str, filename: str = "candidates_filtered.csv"
) -> Path:
    out = rule_selection_dir(root, log_name) / filename
    df.to_csv(out, index=False)
    return out


def write_ranking(df: pd.DataFrame, root: Path, log_name: str) -> Path:
    out = rule_selection_dir(root, log_name) / "uplift_ranking.csv"
    df.to_csv(out, index=False)
    return out


def write_screening_summary(df: pd.DataFrame, root: Path, log_name: str) -> Path:
    out = rule_selection_dir(root, log_name) / "screening_summary.csv"
    df.to_csv(out, index=False)
    return out


def write_final_comparison(df: pd.DataFrame, root: Path, log_name: str) -> Path:
    out = rule_selection_dir(root, log_name) / "final_comparison.csv"
    df.to_csv(out, index=False)
    return out
