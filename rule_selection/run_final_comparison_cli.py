"""
Programa modular #3: compara los 4 brazos (AS-IS, Placebo, Top-Support, GENESIS)
con el mismo numero de replicas para los 4 (CLAUDE.md 12.6) y calcula la
descomposicion causal Delta_generador / Delta_regla.

Requiere haber corrido antes el programa #2 (`run_screening.py`) para este log --
reutiliza su `screening_summary.csv` (para saber cual candidata gano y reusar su BPS
ya descubierto, sin re-alucinar) y su `candidates_supported.csv` (para calcular
Top-Support sobre el mismo pool de candidatas, sin volver a correr MINERful).

Ejemplos (correr desde la raiz del repo, con el env `deep_generator` activo):

    # Corrida rigurosa: poblacion real del log, 20 replicas por brazo (R_final)
    python -m rule_selection.run_final_comparison_cli --log RunningExample.csv ^
        --asis-run-label asis_screening --replicas 20

    # Corrida rapida de supervision
    python -m rule_selection.run_final_comparison_cli --log RunningExample.csv ^
        --asis-run-label asis_screening --replicas 2 ^
        --hallucination-cases 100 --tobe-cases 100

Salida: data/5.rule_selection/<log>/final_comparison.csv.

IMPORTANTE (ver CLAUDE.md 22.13): `event_log_predictor.py` usa `multiprocessing.Pool`,
que en Windows usa `spawn` -- por eso todo el cuerpo corre dentro de
`if __name__ == "__main__":`. No importar/llamar `main()` desde un script sin ese
mismo guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.run_final_comparison_cli")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. RunningExample.csv)")
    parser.add_argument(
        "--replicas", type=int, default=20, help="R_final -- mismo numero de replicas para los 4 brazos"
    )
    parser.add_argument(
        "--hallucination-cases",
        type=int,
        default=None,
        help="Trazas a alucinar por brazo (default: poblacion real del log)",
    )
    parser.add_argument(
        "--tobe-cases",
        type=int,
        default=None,
        help="Casos a simular en Prosimos por brazo (default: poblacion real del log)",
    )
    parser.add_argument(
        "--screening-summary",
        default=None,
        help="Default: data/5.rule_selection/<log>/screening_summary.csv (salida de run_screening.py)",
    )
    parser.add_argument(
        "--candidates-csv",
        default=None,
        help="Default: data/5.rule_selection/<log>/candidates_supported.csv (salida de run_screening.py)",
    )
    parser.add_argument(
        "--asis-run-label",
        required=True,
        help="AS-IS ya descubierto por run_screening.py (ej. asis_screening, o el que se haya usado)",
    )
    parser.add_argument(
        "--tobe-config-filename",
        default="configuration_generated.yaml",
        help="Archivo de configuracion de Simod para Placebo y Top-Support, debe existir en "
        "data/2.hallucination_logs/<log>/ (default: configuration_generated.yaml). GENESIS usa "
        "la configuracion que ya se uso al descubrirlo en run_screening.py -- no aplica aqui.",
    )
    parser.add_argument(
        "--neutralize-markers",
        action="store_true",
        help="Aplicar el parche de marcadores sinteticos (CLAUDE.md 22.24/22.27) a Placebo y "
        "Top-Support. AS-IS reutiliza --asis-run-label tal cual (aplicar el parche antes, al "
        "descubrirlo). GENESIS usa lo que ya se aplico en run_screening.py.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    import pandas as pd

    from params.Params import Params
    from rule_selection import artifacts
    from rule_selection.final_comparison import run_final_comparison
    from rule_selection.simulation_arms import reconstruct_tobe_handle

    log_name = args.log.replace(".csv", "")
    base_params = Params(
        root=root,
        log_filename=args.log,
        hallucination_cases=args.hallucination_cases,
        tobe_cases=args.tobe_cases,
    )
    print(f"Log: {args.log} | poblacion real: {base_params.total_cases} casos")
    print(f"hallucination_cases (target LSTM, por brazo): {args.hallucination_cases or base_params.total_cases}")
    print(f"tobe_cases efectivo (Prosimos, los 4 brazos): {base_params.effective_tobe_cases}")

    screening_summary_path = (
        Path(args.screening_summary)
        if args.screening_summary
        else artifacts.rule_selection_dir(root, log_name) / "screening_summary.csv"
    )
    candidates_path = (
        Path(args.candidates_csv)
        if args.candidates_csv
        else artifacts.rule_selection_dir(root, log_name) / "candidates_supported.csv"
    )
    if not screening_summary_path.exists():
        raise FileNotFoundError(
            f"No existe {screening_summary_path} -- corre primero run_screening.py para este log."
        )
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"No existe {candidates_path} -- corre primero run_screening.py para este log."
        )

    screening_summary = pd.read_csv(screening_summary_path)
    df_candidates = pd.read_csv(candidates_path)
    winner = screening_summary.iloc[0]
    winner_run_label = f"screening_{winner['candidate_id']}"
    print(f"\nGanadora del screening: {winner['candidate_id']} (PCE screening = {winner['screening_mean_pce']:.2f})")

    genesis_params = Params(
        root=root,
        log_filename=args.log,
        rules_filename=f"{winner['candidate_id']}.ini",
        hallucination_cases=args.hallucination_cases,
        tobe_cases=args.tobe_cases,
    )
    genesis_handle = reconstruct_tobe_handle(genesis_params, winner_run_label)
    print(
        f"BPS reutilizado: {genesis_handle.bps_folder} "
        f"({len(genesis_handle.stats_csv_paths)} replicas ya corridas en el screening)"
    )
    if not genesis_handle.bps_folder.exists():
        raise FileNotFoundError(
            f"No existe {genesis_handle.bps_folder} -- confirma que run_screening.py corrio "
            "esta candidata (revisa screening_summary.csv)."
        )

    bps_asis_folder = base_params.routes["bps_asis"] / args.asis_run_label
    if not bps_asis_folder.exists():
        raise FileNotFoundError(f"No existe {bps_asis_folder}")

    print(f"Configuracion TO-BE (Placebo/Top-Support): {args.tobe_config_filename}")
    print(f"\n=== Corriendo comparacion final: 4 brazos x {args.replicas} replicas ===\n")
    final_df = run_final_comparison(
        base_params,
        df_candidates,
        winner,
        genesis_handle,
        bps_asis_folder=bps_asis_folder,
        root_path=root,
        replicas=args.replicas,
        tobe_configuration_filename=args.tobe_config_filename,
        neutralize_instantaneous_resources=args.neutralize_markers,
    )

    out = artifacts.write_final_comparison(final_df, root, log_name)
    print(f"\n=== RESULTADO (guardado en {out}) ===")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
