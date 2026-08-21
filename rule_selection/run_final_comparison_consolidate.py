"""
Programa modular #3c: consolida las 4 filas de `final_comparison_partial.csv`
(escritas una por una por `run_final_arm.py`) en el `final_comparison.csv` final,
calculando la descomposicion causal (CLAUDE.md 12.6):

    Delta_generador           = Placebo - AS-IS
    Delta_regla(GENESIS)       = GENESIS - Placebo
    Delta_regla(Top-Support)   = Top-Support - Placebo

Mismo formato de salida que `run_final_comparison_cli.py` (una sola corrida, 4
brazos en un proceso) -- esto es solo la version segmentada, columna por columna
identica.

Ejemplo:

    python -m rule_selection.run_final_comparison_consolidate --log BPI_Challenge_2012.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.run_final_comparison_consolidate")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. BPI_Challenge_2012.csv)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    import pandas as pd

    from rule_selection import artifacts

    log_name = args.log.replace(".csv", "")
    partial_path = artifacts.rule_selection_dir(root, log_name) / "final_comparison_partial.csv"
    if not partial_path.exists():
        raise FileNotFoundError(
            f"No existe {partial_path} -- corre run_final_arm.py para los 4 brazos primero."
        )

    partial = pd.read_csv(partial_path)
    expected = {"AS-IS", "Placebo", "Top-Support", "GENESIS"}
    present = set(partial["brazo"])
    missing = expected - present
    if missing:
        raise RuntimeError(
            f"Faltan brazos en {partial_path}: {sorted(missing)} -- corre run_final_arm.py "
            f"con --arm para cada uno antes de consolidar."
        )

    mean_pce = dict(zip(partial["brazo"], partial["mean_PCE"]))
    n_replicas = dict(zip(partial["brazo"], partial["n_replicas"]))

    delta_generador = mean_pce["Placebo"] - mean_pce["AS-IS"]
    delta_regla_genesis = mean_pce["GENESIS"] - mean_pce["Placebo"]
    delta_regla_topsupport = mean_pce["Top-Support"] - mean_pce["Placebo"]

    rows = [
        {
            "brazo": "AS-IS",
            "mean_PCE": mean_pce["AS-IS"],
            "n_replicas": n_replicas["AS-IS"],
            "delta_generador": None,
            "delta_regla": None,
        },
        {
            "brazo": "Placebo",
            "mean_PCE": mean_pce["Placebo"],
            "n_replicas": n_replicas["Placebo"],
            "delta_generador": delta_generador,
            "delta_regla": None,
        },
        {
            "brazo": "Top-Support",
            "mean_PCE": mean_pce["Top-Support"],
            "n_replicas": n_replicas["Top-Support"],
            "delta_generador": None,
            "delta_regla": delta_regla_topsupport,
        },
        {
            "brazo": "GENESIS",
            "mean_PCE": mean_pce["GENESIS"],
            "n_replicas": n_replicas["GENESIS"],
            "delta_generador": None,
            "delta_regla": delta_regla_genesis,
        },
    ]
    final_df = pd.DataFrame(rows)
    out = artifacts.write_final_comparison(final_df, root, log_name)
    print(f"=== RESULTADO (guardado en {out}) ===")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
