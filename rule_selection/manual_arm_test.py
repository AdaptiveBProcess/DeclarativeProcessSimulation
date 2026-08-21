"""
Prueba manual de UN brazo de simulacion (una candidata): alucinacion LSTM -> Simod -> Prosimos.

Util para supervisar que el pipeline funcione de punta a punta sin esperar a la corrida
completa (screening de 5 candidatas + comparacion final de 4 brazos, CLAUDE.md 12.5-12.7).

Ejemplos (correr desde la raiz del repo, con el env `deep_generator` activo):

    # Corrida rapida de supervision (100 casos en vez de la poblacion completa del log)
    python -m rule_selection.manual_arm_test --log RunningExample.csv ^
        --rule notchainsuccession_Task_C_Task_E.ini --replicas 2 ^
        --hallucination-cases 100 --tobe-cases 100

    # Corrida rigurosa (poblacion real del log completo, ej. 540 en RunningExample --
    # simplemente se omiten --hallucination-cases/--tobe-cases)
    python -m rule_selection.manual_arm_test --log RunningExample.csv ^
        --rule notchainsuccession_Task_C_Task_E.ini --replicas 2

    # Reutilizar un AS-IS ya descubierto en vez de correr Simod de nuevo sobre el log original
    python -m rule_selection.manual_arm_test --log RunningExample.csv ^
        --rule notchainsuccession_Task_C_Task_E.ini --asis-run-label asis_smoketest_v3

IMPORTANTE (ver CLAUDE.md 22.13): `event_log_predictor.py` usa `multiprocessing.Pool`, que en
Windows usa `spawn` -- reimporta el modulo `__main__` en cada proceso hijo. Por eso este archivo
llama a `main()` dentro de `if __name__ == "__main__":` -- sin ese guard, cada proceso hijo
re-ejecutaria el script completo desde cero, incluida la llamada a `run_tobe_arm`, causando una
explosion de procesos en cascada (ya ocurrio una vez en esta sesion: un script sin el guard
genero un log de 344.000+ lineas en minutos antes de poder detenerlo).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.manual_arm_test")
    parser.add_argument("--root", default=".", help="Raiz del proyecto (default: directorio actual)")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. RunningExample.csv)")
    parser.add_argument(
        "--rule",
        required=True,
        help="Nombre del archivo .ini de la candidata, dentro de data/0.logs/<log>/ "
        "(ej. notchainsuccession_Task_C_Task_E.ini)",
    )
    parser.add_argument(
        "--run-label", default=None, help="Etiqueta de la corrida (default: derivada del nombre de la regla)"
    )
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument(
        "--hallucination-cases",
        type=int,
        default=None,
        help="Trazas a alucinar con el LSTM (default: poblacion real del log)",
    )
    parser.add_argument(
        "--tobe-cases",
        type=int,
        default=None,
        help="Casos a simular en Prosimos, AS-IS y TO-BE por igual (default: poblacion real del log)",
    )
    parser.add_argument(
        "--asis-run-label",
        default=None,
        help="Reutilizar un AS-IS ya descubierto en data/3.bps_asis/<log>/<label> "
        "en vez de correr Simod de nuevo sobre el log original",
    )
    parser.add_argument(
        "--asis-config-filename",
        default="configuration_original.yaml",
        help="Solo si se descubre AS-IS de nuevo (sin --asis-run-label): archivo de "
        "configuracion de Simod, debe existir en data/2.input_logs/<log>/",
    )
    parser.add_argument(
        "--tobe-config-filename",
        default="configuration_generated.yaml",
        help="Archivo de configuracion de Simod para el brazo TO-BE, debe existir en "
        "data/2.hallucination_logs/<log>/ (default: configuration_generated.yaml)",
    )
    parser.add_argument("--variant", default="Rules Based Random Choice")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    from params.Params import Params
    from rule_selection import simulation_arms as sa
    from rule_selection.kpi import mean_pce_over_replicas

    run_label = args.run_label or f"manual_{Path(args.rule).stem}"

    params = Params(
        root=root,
        log_filename=args.log,
        rep=1,
        variant=args.variant,
        rules_filename=args.rule,
        hallucination_cases=args.hallucination_cases,
        tobe_cases=args.tobe_cases,
    )

    print(f"Log: {args.log} | poblacion real: {params.total_cases} casos")
    print(f"Regla: {args.rule}")
    print(f"hallucination_cases (target LSTM): {args.hallucination_cases or params.total_cases}")
    print(f"tobe_cases efectivo (Prosimos, AS-IS y TO-BE): {params.effective_tobe_cases}")

    if args.asis_run_label:
        bps_asis_folder = params.routes["bps_asis"] / args.asis_run_label
        if not bps_asis_folder.exists():
            raise FileNotFoundError(
                f"No existe {bps_asis_folder} -- omite --asis-run-label para descubrirlo de nuevo."
            )
        print(f"Reutilizando AS-IS ya descubierto: {bps_asis_folder}")
    else:
        print(f"Descubriendo AS-IS con Simod (config={args.asis_config_filename}, no depende de la regla)...")
        bps_asis_folder = sa.discover_asis_bps(
            params, run_label=f"{run_label}_asis", configuration_filename=args.asis_config_filename
        )

    print(f"Configuracion TO-BE: {args.tobe_config_filename}")
    print(f"\n=== Corriendo brazo TO-BE, run_label={run_label}, {args.replicas} replicas ===\n")
    handle = sa.run_tobe_arm(
        params=params,
        replicas=args.replicas,
        run_label=run_label,
        root_path=root,
        bps_asis_folder=bps_asis_folder,
        discover=True,
        configuration_filename=args.tobe_config_filename,
    )

    mean_pce = mean_pce_over_replicas(handle.stats_csv_paths)
    print("\n=== RESULTADO ===")
    print("hallucinated_folder:", handle.hallucinated_folder)
    print(f"PCE medio simulado ({len(handle.stats_csv_paths)} replicas): {mean_pce:.2f}")


if __name__ == "__main__":
    main()
