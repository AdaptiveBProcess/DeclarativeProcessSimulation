"""
Programa modular #2: simula las top-k candidatas seleccionadas por el recomendador
(CLAUDE.md 12.5) -- para cada una, alucinacion LSTM + descubrimiento Simod UNA vez,
mas N replicas de Prosimos.

Ejemplos (correr desde la raiz del repo, con el env `deep_generator` activo):

    # Corrida rigurosa completa: recomputa el ranking (pasos 1-4) y descubre AS-IS
    # de cero, poblacion real del log, 5 replicas por candidata
    python -m rule_selection.run_screening --log RunningExample.csv

    # Corrida rapida de supervision: reutiliza un ranking y un AS-IS ya calculados,
    # 100 casos en vez de la poblacion completa, 2 replicas
    python -m rule_selection.run_screening --log RunningExample.csv ^
        --ranking-csv data/5.rule_selection/RunningExample/uplift_ranking.csv ^
        --asis-run-label asis_smoketest_v3 --replicas 2 ^
        --hallucination-cases 100 --tobe-cases 100

Salida: data/5.rule_selection/<log>/screening_summary.csv (ganadora = mayor PCE
simulado promedio -- criterio confirmado con el usuario, no una segunda aplicacion
del score de Welch). Cada candidata queda en
data/3.bps_tobe/<log>/screening_<candidate_id>/ y
data/4.simulation_results/<log>/screening_<candidate_id>_rep{N}/ -- estas rutas las
reutiliza el programa #3 (run_final_comparison_cli.py) para el brazo GENESIS sin
volver a alucinar/descubrir.

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
    parser = argparse.ArgumentParser(prog="rule_selection.run_screening")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. RunningExample.csv)")
    parser.add_argument(
        "--minerful-root", default=None, help="Carpeta con MINERful.jar/lib (default: <root>/MINERful)"
    )
    parser.add_argument("--s-min", type=float, default=0.05)
    parser.add_argument("--c-min", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=5, help="Replicas de Prosimos por candidata")
    parser.add_argument(
        "--hallucination-cases",
        type=int,
        default=None,
        help="Trazas a alucinar por candidata (default: poblacion real del log)",
    )
    parser.add_argument(
        "--tobe-cases",
        type=int,
        default=None,
        help="Casos a simular en Prosimos por candidata (default: poblacion real del log)",
    )
    parser.add_argument(
        "--ranking-csv",
        default=None,
        help="Reutilizar un ranking ya calculado (ej. uplift_ranking.csv) en vez de "
        "recomputar los pasos 1-4 (candidatas MINERful + uplift + Welch)",
    )
    parser.add_argument(
        "--asis-run-label",
        default=None,
        help="Reutilizar un AS-IS ya descubierto en data/3.bps_asis/<log>/<label> "
        "en vez de correr Simod de nuevo sobre el log original",
    )
    parser.add_argument(
        "--tobe-config-filename",
        default="configuration_generated.yaml",
        help="Archivo de configuracion de Simod para TO-BE, debe existir en "
        "data/2.hallucination_logs/<log>/ (default: configuration_generated.yaml)",
    )
    parser.add_argument(
        "--asis-config-filename",
        default="configuration_original.yaml",
        help="Solo si se descubre AS-IS de nuevo (sin --asis-run-label): archivo de "
        "configuracion de Simod, debe existir en data/2.input_logs/<log>/",
    )
    parser.add_argument(
        "--neutralize-markers",
        action="store_true",
        help="Aplicar el parche de marcadores sinteticos (EVENT 1 START/EVENT 7 END con "
        "recurso inventado, CLAUDE.md 22.24/22.27) a las 5 candidatas -- y a AS-IS si se "
        "descubre de nuevo (sin --asis-run-label).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    import pandas as pd

    from params.Params import Params
    from rule_selection import artifacts
    from rule_selection.ranking_pipeline import compute_ranking
    from rule_selection.screening import run_screening
    from rule_selection.simulation_arms import discover_asis_bps

    minerful_root = Path(args.minerful_root) if args.minerful_root else root / "MINERful"
    log_name = args.log.replace(".csv", "")

    base_params = Params(
        root=root,
        log_filename=args.log,
        hallucination_cases=args.hallucination_cases,
        tobe_cases=args.tobe_cases,
    )
    print(f"Log: {args.log} | poblacion real: {base_params.total_cases} casos")
    print(f"hallucination_cases (target LSTM, por candidata): {args.hallucination_cases or base_params.total_cases}")
    print(f"tobe_cases efectivo (Prosimos, por candidata): {base_params.effective_tobe_cases}")

    if args.ranking_csv:
        print(f"\nReutilizando ranking: {args.ranking_csv}")
        ranked = pd.read_csv(args.ranking_csv)
    else:
        print("\nRecomputando ranking (pasos 1-4: candidatas MINERful + uplift + score de Welch)...")
        _, ranked = compute_ranking(
            root, args.log, minerful_root, s_min=args.s_min, c_min=args.c_min, top_k=args.top_k
        )

    print(f"\nTop-{len(ranked)} candidatas a simular:")
    print(ranked[["candidate_id", "score"]].to_string(index=False))

    if args.asis_run_label:
        bps_asis_folder = base_params.routes["bps_asis"] / args.asis_run_label
        if not bps_asis_folder.exists():
            raise FileNotFoundError(
                f"No existe {bps_asis_folder} -- omite --asis-run-label para descubrirlo de nuevo."
            )
        print(f"\nReutilizando AS-IS ya descubierto: {bps_asis_folder}")
    else:
        print("\nDescubriendo AS-IS con Simod (una sola vez, no depende de ninguna candidata)...")
        bps_asis_folder = discover_asis_bps(
            base_params,
            run_label="asis_screening",
            configuration_filename=args.asis_config_filename,
            neutralize_instantaneous_resources=args.neutralize_markers,
        )

    print(f"Configuracion TO-BE: {args.tobe_config_filename}")
    print(f"Neutralizar marcadores sinteticos: {args.neutralize_markers}")
    print(f"\n=== Corriendo screening: {len(ranked)} candidatas x {args.replicas} replicas ===\n")
    summary, handles = run_screening(
        base_params,
        ranked,
        bps_asis_folder,
        root_path=root,
        replicas=args.replicas,
        tobe_configuration_filename=args.tobe_config_filename,
        neutralize_instantaneous_resources=args.neutralize_markers,
    )

    out = artifacts.write_screening_summary(summary, root, log_name)
    print(f"\n=== RESULTADO (guardado en {out}) ===")
    print(summary.to_string(index=False))
    winner = summary.iloc[0]
    print(f"\nGanadora: {winner['candidate_id']} (PCE medio simulado = {winner['screening_mean_pce']:.2f})")
    print(f"AS-IS usado: {bps_asis_folder} (reutilizable con --asis-run-label {bps_asis_folder.name})")


if __name__ == "__main__":
    main()
