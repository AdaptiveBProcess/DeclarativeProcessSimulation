"""
Entrypoint del pipeline de seleccion de reglas GENESIS (CLAUDE.md Seccion 12).

Compone: candidatas (MINERful una sola vez) -> cumplimiento por traza -> uplift y
score de Welch -> top-5 -> screening por simulacion -> comparacion final de 4
brazos. No usa `cli.py` (confirmado no importable: importa `compress_csv_to_gz` de
`dg_prediction.py`, que no existe ahi). Sigue el patron de `dg_prediction.py::main()`
para la orquestacion de simulacion, sin repetir sus bugs conocidos (ver
`simulation_arms.py`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import dg_training
from params.Params import Params

from rule_selection import artifacts
from rule_selection._util import chdir as _chdir
from rule_selection.final_comparison import run_final_comparison
from rule_selection.ranking_pipeline import compute_ranking
from rule_selection.screening import run_screening, select_winner
from rule_selection.simulation_arms import discover_asis_bps


@dataclass
class PipelineResult:
    candidates: pd.DataFrame
    ranking: pd.DataFrame
    screening_summary: pd.DataFrame
    final_comparison: pd.DataFrame


def train_lstm_once(root: Path, log_filename: str) -> None:
    """
    Entrena el LSTM UNA vez para el log completo -- confirmado rule-independiente
    (sin referencias a Declare/reglas en `GenerativeLSTM/model_training/`), se
    reutiliza para todas las candidatas cambiando solo `rules_path` por corrida
    (`simulation_arms.run_tobe_arm`). `dg_training.py::main()` usa rutas relativas
    fijas (`data/0.logs/{NAME}`) y no acepta una raiz como parametro -- se corre con
    cwd=root.

    ADVERTENCIA: esto entrena un modelo LSTM de verdad (200 epochs, busqueda
    bayesiana de hiperparametros) -- puede tardar mucho. Omitir con
    `train_lstm=False` si ya existe un modelo entrenado para este log.
    """
    with _chdir(root):
        dg_training.main(["-f", log_filename])


def run_pipeline(
    root: Path,
    log_filename: str,
    minerful_root: Path,
    s_min: float = 0.05,
    c_min: float = 0.8,
    top_k_screening: int = 5,
    screening_replicas: int = 5,
    final_replicas: int = 20,
    train_lstm: bool = True,
    hallucination_cases: int | None = None,
    tobe_cases: int | None = None,
) -> PipelineResult:
    root = Path(root)
    log_name = log_filename.replace(".csv", "")

    # `hallucination_cases`/`tobe_cases` en None (default) usan la poblacion real del log
    # (Params.total_cases) -- pasar un numero explicito (ej. 100) da una corrida liviana de
    # supervision, sin esperar a alucinar/simular la poblacion completa.
    base_params = Params(
        root=root,
        log_filename=log_filename,
        hallucination_cases=hallucination_cases,
        tobe_cases=tobe_cases,
    )

    if train_lstm:
        train_lstm_once(root, log_filename)

    # --- Pasos 1-4 (CLAUDE.md 12.2-12.4): candidatas, cumplimiento, uplift, ranking
    # --- No requieren Docker, solo Java para MINERful. Extraido a ranking_pipeline.py
    # --- para que run_screening.py (programa modular independiente) lo reutilice sin
    # --- duplicar esta logica.
    df_candidates, ranked = compute_ranking(
        root, log_filename, minerful_root, s_min=s_min, c_min=c_min, top_k=top_k_screening
    )

    # --- AS-IS: Simod una sola vez, reutilizado por screening y comparacion final ---
    bps_asis_folder = discover_asis_bps(base_params)

    # --- Paso 5 (12.5): screening por simulacion de las top-k candidatas ---
    screening_summary, handles = run_screening(
        base_params, ranked, bps_asis_folder, root_path=root, replicas=screening_replicas
    )
    artifacts.write_screening_summary(screening_summary, root, log_name)
    genesis_winner, genesis_handle = select_winner(screening_summary, handles)

    # --- Paso 6 (12.6): comparacion final de 4 brazos, mismas replicas para todos ---
    final_df = run_final_comparison(
        base_params,
        df_candidates,
        genesis_winner,
        genesis_handle,
        bps_asis_folder=bps_asis_folder,
        root_path=root,
        replicas=final_replicas,
    )
    artifacts.write_final_comparison(final_df, root, log_name)

    return PipelineResult(
        candidates=df_candidates,
        ranking=ranked,
        screening_summary=screening_summary,
        final_comparison=final_df,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.orchestrate")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. RunningExample.csv)")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument(
        "--minerful-root", default=None, help="Carpeta con MINERful.jar/lib (default: <root>/MINERful)"
    )
    parser.add_argument("--s-min", type=float, default=0.05)
    parser.add_argument("--c-min", type=float, default=0.8)
    parser.add_argument("--top-k-screening", type=int, default=5)
    parser.add_argument("--screening-replicas", type=int, default=5)
    parser.add_argument("--final-replicas", type=int, default=20)
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Omitir el entrenamiento del LSTM (reutilizar un modelo ya entrenado para este log)",
    )
    parser.add_argument(
        "--hallucination-cases",
        type=int,
        default=None,
        help="Trazas a alucinar con el LSTM (default: poblacion real del log, ej. 540 en RunningExample). "
        "Usar un numero chico (ej. 100) para una corrida rapida de supervision.",
    )
    parser.add_argument(
        "--tobe-cases",
        type=int,
        default=None,
        help="Casos a simular en Prosimos, AS-IS y TO-BE por igual (default: poblacion real del log). "
        "Usar un numero chico (ej. 100) para una corrida rapida de supervision.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    minerful_root = Path(args.minerful_root) if args.minerful_root else root / "MINERful"

    result = run_pipeline(
        root=root,
        log_filename=args.log,
        minerful_root=minerful_root,
        s_min=args.s_min,
        c_min=args.c_min,
        top_k_screening=args.top_k_screening,
        screening_replicas=args.screening_replicas,
        final_replicas=args.final_replicas,
        train_lstm=not args.no_train,
        hallucination_cases=args.hallucination_cases,
        tobe_cases=args.tobe_cases,
    )

    print("\n=== Ranking de candidatas (top-k por score de Welch) ===")
    print(result.ranking.to_string(index=False))
    print("\n=== Resumen de screening (PCE simulado, 5 candidatas x 5 replicas) ===")
    print(result.screening_summary.to_string(index=False))
    print("\n=== Comparacion final: AS-IS / Placebo / Top-Support / GENESIS ===")
    print(result.final_comparison.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
