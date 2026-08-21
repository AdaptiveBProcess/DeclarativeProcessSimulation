"""
Programa modular: prueba una variante de configuracion de Simod para el
descubrimiento de AS-IS (ej. `discovery_type: undifferentiated` vs `differentiated`
para calendarios de recursos), sin tocar el pipeline de reglas ni el generador LSTM.
Reusable para cualquier experimento futuro de este tipo -- basta con guardar una
nueva variante de `configuration_original.yaml` en `data/2.input_logs/<log>/` y
pasar su nombre por `--config-filename`.

Contexto (CLAUDE.md 22.19-22.20): AS-IS simulado con la config original
(`differentiated`) da PCE~0.97% vs ~100-104% del log real -- causa raiz
identificada: calendarios de recursos fragmentados en bloques de ~1h (15.8 h/semana
promedio de 40 posibles, sobre 31 calendarios).

Ejemplo (correr desde la raiz del repo, con el env `deep_generator` activo):

    python -m rule_selection.asis_config_test --log RunningExample.csv ^
        --config-filename configuration_original_undifferentiated.yaml ^
        --run-label asis_undifferentiated --replicas 2

No usa `multiprocessing.Pool` (a diferencia de los programas que alucinan con el
LSTM) -- AS-IS no alucina, así que no aplica el riesgo de fork bomb de CLAUDE.md
22.13. Se mantiene el guard `if __name__ == "__main__":` por consistencia con el
resto de `rule_selection/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.asis_config_test")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. RunningExample.csv)")
    parser.add_argument(
        "--config-filename",
        default="configuration_original.yaml",
        help="Archivo de configuracion de Simod, debe existir en data/2.input_logs/<log>/",
    )
    parser.add_argument(
        "--run-label", required=True, help="Namespace para el BPS descubierto y las replicas de esta variante"
    )
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument(
        "--tobe-cases", type=int, default=None, help="Casos a simular en Prosimos (default: poblacion real del log)"
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    from params.Params import Params
    from rule_selection import simulation_arms as sa
    from rule_selection.calendar_diagnostics import find_bps_model_json, summarize_resource_calendars
    from rule_selection.kpi import mean_pce_over_replicas

    params = Params(root=root, log_filename=args.log, tobe_cases=args.tobe_cases)
    print(f"Log: {args.log} | poblacion real: {params.total_cases} casos")
    print(f"Configuracion Simod: {args.config_filename}")
    print(f"tobe_cases efectivo (Prosimos): {params.effective_tobe_cases}")

    print(f"\nDescubriendo AS-IS con Simod (run_label={args.run_label})...")
    bps_asis_folder = sa.discover_asis_bps(
        params, run_label=args.run_label, configuration_filename=args.config_filename
    )

    model_json = find_bps_model_json(bps_asis_folder, params.name)
    if model_json:
        cal_summary = summarize_resource_calendars(model_json)
        print(f"\n=== Calendarios de recursos descubiertos ({len(cal_summary)}) ===")
        print(f"Horas/semana -- media: {cal_summary['weekly_hours'].mean():.1f}  "
              f"mediana: {cal_summary['weekly_hours'].median():.1f}  "
              f"min: {cal_summary['weekly_hours'].min():.1f}  "
              f"max: {cal_summary['weekly_hours'].max():.1f}  "
              f"(referencia: semana completa Lun-Vie 9-17 = 40 h/semana)")
    else:
        print("\n(no se encontro el .json del modelo -- se omite el diagnostico de calendarios)")

    print(f"\n=== Simulando {args.replicas} replicas ===\n")
    handle = sa.run_asis_arm(params, bps_asis_folder=bps_asis_folder, replicas=args.replicas, run_label=args.run_label)

    mean_pce = mean_pce_over_replicas(handle.stats_csv_paths)
    print("\n=== RESULTADO ===")
    print(f"Config: {args.config_filename}")
    print(f"PCE medio simulado AS-IS ({args.replicas} replicas): {mean_pce:.2f}")


if __name__ == "__main__":
    main()
