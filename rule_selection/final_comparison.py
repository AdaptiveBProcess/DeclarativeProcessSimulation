"""
Paso 6 (CLAUDE.md 12.6): comparacion final de 4 brazos, con el mismo numero de
replicas (`R_final`) para cada uno -- corrige la asimetria de replicas del pipeline
legacy (AS-IS se simulaba 1 sola vez, TO-BE 3 veces).

Los 4 brazos:
  - AS-IS:        log real, sin alucinacion, `R_final` replicas de Prosimos.
  - Placebo:      alucinacion SIN restriccion (`variant="Random Choice"`) + Simod +
                   `R_final` replicas. Aisla el efecto del generador solo.
  - Top-Support:  regla de mayor Support+Coverage+Confidence del MISMO pool de
                   candidatas que genero `candidates.py` (sin volver a correr
                   MINERful) + alucinacion restringida + Simod + `R_final` replicas.
  - GENESIS:      la ganadora del screening -- reutiliza el BPS ya descubierto ahi
                   (`discover=False`), solo corre las replicas que faltan para
                   llegar a `R_final`.

Descomposicion causal (responde a la pregunta del revisor "how can a policy derived
from the same methodology be a fair benchmark?"):
    Delta_generador           = Placebo - AS-IS
    Delta_regla(GENESIS)       = GENESIS - Placebo
    Delta_regla(Top-Support)   = Top-Support - Placebo
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from params.Params import Params
from rule_selection.candidates import top_support_candidate
from rule_selection.kpi import mean_pce_over_replicas
from rule_selection.simulation_arms import (
    ArmRunHandle,
    make_candidate_params,
    run_asis_arm,
    run_tobe_arm,
)


def _clean_arg1(value):
    return None if pd.isna(value) else value


def run_final_comparison(
    base_params: Params,
    df_candidates: pd.DataFrame,
    genesis_winner: pd.Series,
    genesis_handle: ArmRunHandle,
    bps_asis_folder: Path,
    root_path: Path,
    replicas: int = 20,
    tobe_configuration_filename: str = "configuration_generated.yaml",
    neutralize_instantaneous_resources: bool = False,
) -> pd.DataFrame:
    """
    Corre los 4 brazos con `replicas` corridas de Prosimos cada uno (mismo numero
    para los 4) y calcula la descomposicion causal Delta_generador / Delta_regla.

    `tobe_configuration_filename`/`neutralize_instantaneous_resources` se pasan a
    AS-IS, Placebo y Top-Support (deben existir en `data/2.hallucination_logs/<log>/`,
    ver `simulation_arms.run_tobe_arm`) -- para GENESIS no aplica aqui:
    `genesis_handle` ya trae su propio BPS descubierto (por el screening previo), con
    la configuracion que se haya usado en ESA corrida. Si se quiere GENESIS con una
    configuracion distinta, hay que re-correr el screening con las mismas opciones
    antes de llamar a esta funcion (CLAUDE.md 22.26-22.27).
    """
    # --- AS-IS ---
    asis_handle = run_asis_arm(
        base_params, bps_asis_folder=bps_asis_folder, replicas=replicas, run_label="final_asis"
    )

    # --- Top-Support (mismo pool de candidatas, sin re-correr MINERful) ---
    top_support = top_support_candidate(df_candidates)
    top_support_params = make_candidate_params(
        base_params,
        candidate_id="final_topsupport",
        template=top_support["template"],
        arg0=top_support["arg0"],
        arg1=_clean_arg1(top_support.get("arg1")),
        variant="Rules Based Random Choice",
    )
    topsupport_handle = run_tobe_arm(
        top_support_params,
        replicas=replicas,
        run_label="final_topsupport",
        root_path=root_path,
        bps_asis_folder=bps_asis_folder,
        discover=True,
        configuration_filename=tobe_configuration_filename,
        neutralize_instantaneous_resources=neutralize_instantaneous_resources,
    )

    # --- Placebo: sin restriccion, pero igual necesita un .ini sintacticamente
    # valido (ModelPredictor llama extract_rules incondicionalmente -- ver
    # simulation_arms.py). Se reutiliza el contenido de Top-Support porque es
    # inerte para variant="Random Choice" (no filtra nada durante la generacion).
    placebo_params = make_candidate_params(
        base_params,
        candidate_id="final_placebo",
        template=top_support["template"],
        arg0=top_support["arg0"],
        arg1=_clean_arg1(top_support.get("arg1")),
        variant="Random Choice",
    )
    placebo_handle = run_tobe_arm(
        placebo_params,
        replicas=replicas,
        run_label="final_placebo",
        root_path=root_path,
        bps_asis_folder=bps_asis_folder,
        discover=True,
        configuration_filename=tobe_configuration_filename,
        neutralize_instantaneous_resources=neutralize_instantaneous_resources,
    )

    # --- GENESIS: reutiliza el BPS ya descubierto en el screening, solo le faltan
    # (replicas - replicas_ya_corridas_en_screening) corridas de Prosimos.
    remaining = max(replicas - len(genesis_handle.stats_csv_paths), 0)
    if remaining > 0:
        genesis_handle = run_tobe_arm(
            genesis_handle.params,
            replicas=remaining,
            run_label=genesis_handle.label,
            root_path=root_path,
            discover=False,
            existing_handle=genesis_handle,
        )
    genesis_stats_paths = genesis_handle.stats_csv_paths[:replicas]

    mean_pce = {
        "AS-IS": mean_pce_over_replicas(asis_handle.stats_csv_paths[:replicas]),
        "Placebo": mean_pce_over_replicas(placebo_handle.stats_csv_paths[:replicas]),
        "Top-Support": mean_pce_over_replicas(topsupport_handle.stats_csv_paths[:replicas]),
        "GENESIS": mean_pce_over_replicas(genesis_stats_paths),
    }

    delta_generador = mean_pce["Placebo"] - mean_pce["AS-IS"]
    delta_regla_genesis = mean_pce["GENESIS"] - mean_pce["Placebo"]
    delta_regla_topsupport = mean_pce["Top-Support"] - mean_pce["Placebo"]

    rows = [
        {
            "brazo": "AS-IS",
            "mean_PCE": mean_pce["AS-IS"],
            "n_replicas": len(asis_handle.stats_csv_paths[:replicas]),
            "delta_generador": None,
            "delta_regla": None,
        },
        {
            "brazo": "Placebo",
            "mean_PCE": mean_pce["Placebo"],
            "n_replicas": len(placebo_handle.stats_csv_paths[:replicas]),
            "delta_generador": delta_generador,
            "delta_regla": None,
        },
        {
            "brazo": "Top-Support",
            "mean_PCE": mean_pce["Top-Support"],
            "n_replicas": len(topsupport_handle.stats_csv_paths[:replicas]),
            "delta_generador": None,
            "delta_regla": delta_regla_topsupport,
        },
        {
            "brazo": "GENESIS",
            "mean_PCE": mean_pce["GENESIS"],
            "n_replicas": len(genesis_stats_paths),
            "delta_generador": None,
            "delta_regla": delta_regla_genesis,
        },
    ]
    return pd.DataFrame(rows)
