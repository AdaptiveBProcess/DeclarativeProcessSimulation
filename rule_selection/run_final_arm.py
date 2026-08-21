"""
Programa modular #3b: corre UN SOLO brazo de la comparacion final (AS-IS / Placebo /
Top-Support / GENESIS) por invocacion de proceso.

Por que existe (CLAUDE.md 25.4/25.5): `run_final_comparison_cli.py` corre los 4
brazos en un solo proceso Python continuo. Placebo y Top-Support cada uno hace su
propia alucinacion LSTM completa (371 trazas en BPI_Challenge_2012) -- son 2
alucinaciones consecutivas dentro del mismo proceso, el mismo patron que provoco el
`numpy.core._exceptions._ArrayMemoryError` durante el screening (candidata 3/5, con
2 candidatas ya alucinadas antes en el mismo proceso). Correr un brazo por proceso,
como se hizo manualmente para las 5 candidatas del screening, evita acumular ese
riesgo de memoria entre brazos.

Cada invocacion escribe/actualiza UNA fila en
`data/5.rule_selection/<log>/final_comparison_partial.csv` (upsert por nombre de
brazo, no pisa las filas de los otros brazos ya corridos). Cuando las 4 filas estan
completas, correr `run_final_comparison_consolidate.py` para calcular las deltas
causales y escribir `final_comparison.csv` en el mismo formato que
`run_final_comparison_cli.py` habria producido de una sola corrida.

Ejemplos (correr desde la raiz del repo, con el env `deep_generator` activo, un
brazo por invocacion -- esperar a que termine cada uno antes del siguiente):

    python -m rule_selection.run_final_arm --log BPI_Challenge_2012.csv --arm asis ^
        --asis-run-label asis_screening --replicas 20 ^
        --hallucination-cases 371 --tobe-cases 371

    python -m rule_selection.run_final_arm --log BPI_Challenge_2012.csv --arm placebo ^
        --asis-run-label asis_screening --replicas 20 ^
        --tobe-config-filename configuration_generated_undiff_nodelay.yaml ^
        --hallucination-cases 371 --tobe-cases 371

    python -m rule_selection.run_final_arm --log BPI_Challenge_2012.csv --arm topsupport ^
        --asis-run-label asis_screening --replicas 20 ^
        --tobe-config-filename configuration_generated_undiff_nodelay.yaml ^
        --hallucination-cases 371 --tobe-cases 371

    python -m rule_selection.run_final_arm --log BPI_Challenge_2012.csv --arm genesis ^
        --replicas 20 --hallucination-cases 371 --tobe-cases 371

IMPORTANTE (ver CLAUDE.md 22.13): `event_log_predictor.py` usa `multiprocessing.Pool`,
que en Windows usa `spawn` -- por eso todo el cuerpo corre dentro de
`if __name__ == "__main__":`. No importar/llamar `main()` desde un script sin ese
mismo guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ARM_DISPLAY_NAMES = {
    "asis": "AS-IS",
    "placebo": "Placebo",
    "topsupport": "Top-Support",
    "genesis": "GENESIS",
}


def _clean_arg1(value):
    import pandas as pd

    return None if pd.isna(value) else value


def _upsert_partial_row(partial_path: Path, brazo: str, mean_pce: float, n_replicas: int) -> None:
    import pandas as pd

    if partial_path.exists():
        df = pd.read_csv(partial_path)
        df = df[df["brazo"] != brazo]
    else:
        df = pd.DataFrame(columns=["brazo", "mean_PCE", "n_replicas"])

    new_row = pd.DataFrame([{"brazo": brazo, "mean_PCE": mean_pce, "n_replicas": n_replicas}])
    df = pd.concat([df, new_row], ignore_index=True)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(partial_path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.run_final_arm")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. BPI_Challenge_2012.csv)")
    parser.add_argument(
        "--arm", required=True, choices=list(ARM_DISPLAY_NAMES), help="Que brazo correr en esta invocacion"
    )
    parser.add_argument(
        "--replicas", type=int, default=20, help="R_final -- mismo numero para los 4 brazos"
    )
    parser.add_argument(
        "--hallucination-cases",
        type=int,
        default=None,
        help="Trazas a alucinar (default: poblacion real del log) -- ignorado para --arm asis",
    )
    parser.add_argument(
        "--tobe-cases",
        type=int,
        default=None,
        help="Casos a simular en Prosimos (default: poblacion real del log)",
    )
    parser.add_argument(
        "--asis-run-label",
        default=None,
        help="AS-IS ya descubierto (ej. asis_screening) -- requerido para --arm asis/placebo/topsupport",
    )
    parser.add_argument(
        "--tobe-config-filename",
        default="configuration_generated.yaml",
        help="Config de Simod para Placebo/Top-Support, debe existir en data/2.hallucination_logs/<log>/",
    )
    parser.add_argument(
        "--screening-summary",
        default=None,
        help="Default: data/5.rule_selection/<log>/screening_summary.csv -- solo para --arm genesis",
    )
    parser.add_argument(
        "--candidates-csv",
        default=None,
        help="Default: data/5.rule_selection/<log>/candidates_supported.csv -- solo para "
        "--arm placebo/topsupport (fuente del pool de Top-Support)",
    )
    parser.add_argument(
        "--neutralize-markers",
        action="store_true",
        help="Aplicar el parche de marcadores sinteticos (CLAUDE.md 22.24/22.27) -- solo "
        "para --arm placebo/topsupport (AS-IS y GENESIS reutilizan lo ya descubierto).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    import pandas as pd

    from params.Params import Params
    from rule_selection import artifacts
    from rule_selection.candidates import top_support_candidate
    from rule_selection.kpi import mean_pce_over_replicas
    from rule_selection.simulation_arms import (
        make_candidate_params,
        reconstruct_tobe_handle,
        run_asis_arm,
        run_tobe_arm,
    )

    log_name = args.log.replace(".csv", "")
    base_params = Params(
        root=root,
        log_filename=args.log,
        hallucination_cases=args.hallucination_cases,
        tobe_cases=args.tobe_cases,
    )
    print(f"Log: {args.log} | poblacion real: {base_params.total_cases} casos | brazo: {args.arm}")
    print(f"tobe_cases efectivo (Prosimos): {base_params.effective_tobe_cases}")

    display_name = ARM_DISPLAY_NAMES[args.arm]

    if args.arm in ("asis", "placebo", "topsupport"):
        if not args.asis_run_label:
            raise ValueError(f"--arm {args.arm} requiere --asis-run-label")
        bps_asis_folder = base_params.routes["bps_asis"] / args.asis_run_label
        if not bps_asis_folder.exists():
            raise FileNotFoundError(f"No existe {bps_asis_folder}")

    if args.arm == "asis":
        handle = run_asis_arm(base_params, bps_asis_folder=bps_asis_folder, replicas=args.replicas, run_label="final_asis")
        mean_pce = handle.mean_pce
        n_replicas = len(handle.stats_csv_paths)

    elif args.arm in ("placebo", "topsupport"):
        candidates_path = (
            Path(args.candidates_csv)
            if args.candidates_csv
            else artifacts.rule_selection_dir(root, log_name) / "candidates_supported.csv"
        )
        if not candidates_path.exists():
            raise FileNotFoundError(f"No existe {candidates_path}")
        df_candidates = pd.read_csv(candidates_path)
        top_support = top_support_candidate(df_candidates)
        print(
            f"Regla Top-Support (fuente para Placebo/Top-Support): "
            f"{top_support['template']}({top_support['arg0']}, {_clean_arg1(top_support.get('arg1'))})"
        )

        variant = "Random Choice" if args.arm == "placebo" else "Rules Based Random Choice"
        candidate_id = f"final_{args.arm}"
        candidate_params = make_candidate_params(
            base_params,
            candidate_id=candidate_id,
            template=top_support["template"],
            arg0=top_support["arg0"],
            arg1=_clean_arg1(top_support.get("arg1")),
            variant=variant,
        )
        handle = run_tobe_arm(
            candidate_params,
            replicas=args.replicas,
            run_label=candidate_id,
            root_path=root,
            bps_asis_folder=bps_asis_folder,
            discover=True,
            configuration_filename=args.tobe_config_filename,
            neutralize_instantaneous_resources=args.neutralize_markers,
        )
        mean_pce = handle.mean_pce
        n_replicas = len(handle.stats_csv_paths)

    else:  # genesis
        screening_summary_path = (
            Path(args.screening_summary)
            if args.screening_summary
            else artifacts.rule_selection_dir(root, log_name) / "screening_summary.csv"
        )
        if not screening_summary_path.exists():
            raise FileNotFoundError(f"No existe {screening_summary_path}")
        screening_summary = pd.read_csv(screening_summary_path)
        winner = screening_summary.iloc[0]
        winner_run_label = f"screening_{winner['candidate_id']}"
        print(f"Ganadora del screening: {winner['candidate_id']} (PCE screening = {winner['screening_mean_pce']:.2f})")

        genesis_params = Params(
            root=root,
            log_filename=args.log,
            rules_filename=f"{winner['candidate_id']}.ini",
            hallucination_cases=args.hallucination_cases,
            tobe_cases=args.tobe_cases,
        )
        genesis_handle = reconstruct_tobe_handle(genesis_params, winner_run_label)
        if not genesis_handle.bps_folder.exists():
            raise FileNotFoundError(
                f"No existe {genesis_handle.bps_folder} -- confirma que el screening ya corrio esta candidata."
            )
        print(f"BPS reutilizado: {genesis_handle.bps_folder} ({len(genesis_handle.stats_csv_paths)} replicas ya corridas)")

        remaining = max(args.replicas - len(genesis_handle.stats_csv_paths), 0)
        print(f"Replicas adicionales a correr: {remaining}")
        if remaining > 0:
            genesis_handle = run_tobe_arm(
                genesis_handle.params,
                replicas=remaining,
                run_label=genesis_handle.label,
                root_path=root,
                discover=False,
                existing_handle=genesis_handle,
            )
        stats_paths = genesis_handle.stats_csv_paths[: args.replicas]
        mean_pce = mean_pce_over_replicas(stats_paths)
        n_replicas = len(stats_paths)

    partial_path = artifacts.rule_selection_dir(root, log_name) / "final_comparison_partial.csv"
    _upsert_partial_row(partial_path, display_name, mean_pce, n_replicas)

    print(f"\n=== RESULTADO: {display_name} ===")
    print(f"PCE medio simulado ({n_replicas} replicas): {mean_pce:.2f}")
    print(f"Guardado en {partial_path}")


if __name__ == "__main__":
    main()
