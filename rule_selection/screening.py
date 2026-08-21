"""
Paso 5 (CLAUDE.md 12.5): screening por simulacion de las 5 mejores candidatas.

Para cada una de las top-5 candidatas (por score de Welch, ver `uplift.py`), se
corre el pipeline completo de simulacion (alucinacion -> Simod -> Prosimos) UNA vez,
y solo la simulacion Prosimos se repite 5 veces (25 corridas de Prosimos en total,
pero solo 5 ciclos de alucinacion+descubrimiento Simod -- ver `simulation_arms.py`).
La ganadora es la de mayor PCE simulado promedio (criterio confirmado con el
usuario: media simple entre las 5 replicas, no una segunda aplicacion de la formula
de Welch).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from params.Params import Params
from rule_selection import artifacts
from rule_selection.simulation_arms import ArmRunHandle, make_candidate_params, run_tobe_arm


def _clean_arg1(value):
    return None if pd.isna(value) else value


def run_screening(
    base_params: Params,
    ranked_candidates: pd.DataFrame,
    bps_asis_folder: Path,
    root_path: Path,
    replicas: int = 5,
    tobe_configuration_filename: str = "configuration_generated.yaml",
    neutralize_instantaneous_resources: bool = False,
) -> tuple[pd.DataFrame, dict[str, ArmRunHandle]]:
    """
    Corre el screening sobre las candidatas de `ranked_candidates` (salida de
    `uplift.rank_candidates`, ya limitada a las top-5).

    `tobe_configuration_filename` se pasa a cada candidata (debe existir en
    `data/2.hallucination_logs/<log>/`, ver `simulation_arms.run_tobe_arm`) -- para
    probar una variante de configuracion de Simod (ej. calendarios de recursos,
    CLAUDE.md 22.20-22.26) en las 5 candidatas del screening. `neutralize_instantaneous_resources=True`
    aplica el parche de marcadores sinteticos (CLAUDE.md 22.27) a cada una.

    Devuelve `(resumen, handles)`: `resumen` es la tabla ordenada por PCE simulado
    promedio, y `handles` mapea `candidate_id -> ArmRunHandle` para poder reutilizar
    el BPS ya descubierto de la ganadora en la comparacion final (§12.5: "GENESIS ...
    reutiliza esa simulacion"), sin volver a alucinar/descubrir para ella.

    Imprime progreso por candidata y reescribe `screening_summary.csv` (via
    `artifacts.write_screening_summary`) despues de CADA candidata completada, no
    solo al final -- para una corrida larga sin supervision (varias candidatas x
    varias replicas), si el proceso se corta a mitad de camino (ej. candidata 4 de
    5), el CSV en disco ya tiene el resumen de las candidatas que si terminaron, en
    vez de no tener ningun resultado hasta que las N candidatas completen.
    """
    rows = []
    handles: dict[str, ArmRunHandle] = {}
    log_name = base_params.name
    total = len(ranked_candidates)

    for position, (_, candidate) in enumerate(ranked_candidates.iterrows(), start=1):
        candidate_id = candidate["candidate_id"]
        run_label = f"screening_{candidate_id}"
        score = candidate.get("score")
        print(f"\n--- Regla {position}/{total}: {candidate_id} (score={score:.2f}) ---")

        candidate_params = make_candidate_params(
            base_params,
            candidate_id=candidate_id,
            template=candidate["template"],
            arg0=candidate["arg0"],
            arg1=_clean_arg1(candidate.get("arg1")),
        )

        handle = run_tobe_arm(
            candidate_params,
            replicas=replicas,
            run_label=run_label,
            root_path=root_path,
            bps_asis_folder=bps_asis_folder,
            discover=True,
            configuration_filename=tobe_configuration_filename,
            neutralize_instantaneous_resources=neutralize_instantaneous_resources,
        )
        handles[candidate_id] = handle

        mean_pce = handle.mean_pce
        rows.append(
            {
                "candidate_id": candidate_id,
                "template": candidate["template"],
                "arg0": candidate["arg0"],
                "arg1": _clean_arg1(candidate.get("arg1")),
                "score": score,
                "screening_mean_pce": mean_pce,
                "n_replicas": len(handle.stats_csv_paths),
            }
        )
        print(
            f"--- Regla {position}/{total} completada: {candidate_id} -- "
            f"PCE medio = {mean_pce:.2f} ({len(handle.stats_csv_paths)} replicas) ---"
        )

        partial_summary = (
            pd.DataFrame(rows).sort_values("screening_mean_pce", ascending=False).reset_index(drop=True)
        )
        artifacts.write_screening_summary(partial_summary, root_path, log_name)

    summary = pd.DataFrame(rows).sort_values("screening_mean_pce", ascending=False).reset_index(drop=True)
    return summary, handles


def select_winner(
    summary: pd.DataFrame, handles: dict[str, ArmRunHandle]
) -> tuple[pd.Series, ArmRunHandle]:
    """La ganadora: mayor PCE simulado promedio entre las 5 (criterio confirmado, sin Welch de nuevo)."""
    winner_row = summary.iloc[0]
    return winner_row, handles[winner_row["candidate_id"]]
