"""
Pasos 1-4 (CLAUDE.md 12.2-12.4): generacion de candidatas (MINERful una vez) ->
cumplimiento por traza -> uplift y score de Welch -> ranking top-k.

Extraido de `orchestrate.py::run_pipeline` para que el programa modular de
screening (`run_screening.py`) pueda recomputar o reutilizar el ranking sin
duplicar esta logica -- no requiere Docker, solo Java (para MINERful).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from MINERful.modelo_declarativo_enriquecedio_v2 import (
    cargar_csvs_de_carpeta,
    convertir_date,
    convertir_int,
    corregir_lifecycle_incompleto,
    generar_analisis_por_traza,
)

from rule_selection import artifacts
from rule_selection.candidates import generate_candidates, supported_candidates
from rule_selection.trace_index import build_trace_indexes, build_trace_sequences
from rule_selection.uplift import compute_compliance_table, compute_uplift_stats, rank_candidates


def load_normalized_log(root: Path, log_filename: str) -> pd.DataFrame:
    """
    Replica EXACTAMENTE la secuencia de carga que usa
    `MINERful/modelo_declarativo_enriquecedio_v2.py::main()` (lineas ~1016-1023),
    sin modificar ese archivo -- todo lo que sigue (`generar_analisis_por_traza`,
    `generate_candidates`, `trace_index.build_trace_sequences`) asume esta forma
    normalizada (int/date casteados, lifecycle en minuscula, pares start/complete
    completos).
    """
    name = log_filename.replace(".csv", "")
    log_folder = Path(root) / "data" / "0.logs" / name
    df = cargar_csvs_de_carpeta(str(log_folder))
    if df is None:
        raise FileNotFoundError(f"No se encontraron CSVs en {log_folder}")

    convertir_int(["case:concept:name"], df)
    convertir_date(["time:timestamp"], df)
    df["lifecycle:transition"] = df["lifecycle:transition"].str.lower()
    df = corregir_lifecycle_incompleto(df)
    return df


def compute_ranking(
    root: Path,
    log_filename: str,
    minerful_root: Path,
    s_min: float = 0.05,
    c_min: float = 0.8,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve `(df_candidates_soportadas, ranking_top_k)`. Guarda 3 artefactos en
    `data/5.rule_selection/<log>/`: `candidates_all.csv`, `candidates_supported.csv`,
    `uplift_ranking.csv`.
    """
    root = Path(root)
    log_name = log_filename.replace(".csv", "")

    df_log = load_normalized_log(root, log_filename)

    work_dir = artifacts.rule_selection_dir(root, log_name) / "_minerful_work"
    df_candidates_all = generate_candidates(df_log, minerful_root, work_dir, s_min=s_min, c_min=c_min)
    artifacts.write_candidates(df_candidates_all, root, log_name, filename="candidates_all.csv")

    # El resto del pipeline (uplift, screening, comparacion final) solo puede operar
    # sobre las candidatas que el simulador sabe evaluar/imponer HOY -- las demas
    # (ej. AtLeast3, AtMost3) quedan registradas en candidates_all.csv por si se
    # agrega soporte mas adelante, pero no participan del ranking todavia.
    df_candidates = supported_candidates(df_candidates_all)
    artifacts.write_candidates(df_candidates, root, log_name, filename="candidates_supported.csv")

    sequences = build_trace_sequences(df_log)
    trace_indexes = build_trace_indexes(sequences)

    df_pce = generar_analisis_por_traza(df_log)
    pce_by_trace = df_pce.set_index("traza")["PCE"]

    compliance = compute_compliance_table(trace_indexes, df_candidates)
    stats_df = compute_uplift_stats(compliance, pce_by_trace)
    stats_df = stats_df.merge(
        df_candidates[["candidate_id", "template", "arg0", "arg1"]], on="candidate_id", how="left"
    )
    ranked = rank_candidates(stats_df, top_n=top_k)
    artifacts.write_ranking(ranked, root, log_name)

    return df_candidates, ranked
