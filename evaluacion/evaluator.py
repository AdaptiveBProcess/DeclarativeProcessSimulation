"""
Evaluador de metricas BPS: RED, CTD, 2GD, CONF.
Basado en: Graziosi et al. (2024) arXiv:2411.02131

Uso desde linea de comandos
---------------------------
Comparar un log de referencia contra uno simulado:

    python evaluator.py ref.csv sim.csv

Comparar multiples simulaciones TOBE contra la misma referencia:

    python evaluator.py ref.csv tobe_rep1.csv tobe_rep2.csv tobe_rep3.csv

Guardar tabla resumen en CSV:

    python evaluator.py ref.csv tobe_rep*.csv --output metricas.csv

Ajustar umbral de soporte para CONF (default: 0.90):

    python evaluator.py ref.csv sim.csv --support 0.85

Uso como modulo Python
----------------------
    from evaluacion.evaluator import evaluate, evaluate_batch, mine_once

    # Minar reglas una vez y reutilizarlas para todas las simulaciones
    df_ref = load_log("asis_log.csv")
    rules  = mine_once(df_ref, support_threshold=0.90)

    df_batch = evaluate_batch(
        ref_path  = "asis_log.csv",
        sim_paths = ["tobe1.csv", "tobe2.csv", "tobe3.csv"],
        mined_rules = rules,
    )
    print(df_batch.to_string(index=False))
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from evaluacion.metrics import (
        load_log, compute_red, compute_ctd, compute_2gd,
        compute_conf, discover_conf_model, AVAILABLE_RULES,
    )
except ImportError:
    from metrics import (
        load_log, compute_red, compute_ctd, compute_2gd,
        compute_conf, discover_conf_model, AVAILABLE_RULES,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conveniencia: minar reglas una sola vez
# ─────────────────────────────────────────────────────────────────────────────

def mine_once(df_ref: pd.DataFrame, support_threshold: float = 0.90) -> dict:
    """
    Ejecuta MINERful sobre df_ref y retorna el modelo DECLARE en formato pm4py.
    Util para evaluate_batch: descubrir una vez y reutilizar en todos los lotes.
    """
    cached_model, constraint_strings = discover_conf_model(df_ref, support_threshold)
    return cached_model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluacion individual
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    ref_path: str,
    sim_path: str,
    support_threshold: float = 0.90,
    cached_model: Optional[dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Calcula RED, CTD, 2GD y CONF entre un log de referencia y uno simulado.

    Parametros
    ----------
    ref_path         : ruta al CSV del log de referencia
    sim_path         : ruta al CSV del log simulado (output Prosimos)
    support_threshold: umbral minimo de soporte para CONF (default 0.90)
    cached_model     : modelo pm4py pre-calculado con discover_conf_model
                       (si None, se ejecuta MINERful aqui)
    verbose          : si True, imprime el reporte en consola

    Retorna
    -------
    dict con claves: RED, CTD, CTD_horas, 2GD, CONF, n_rules
    """
    df_ref = load_log(ref_path)
    df_sim = load_log(sim_path)

    n_cases_ref = df_ref["caseid"].nunique()
    n_cases_sim = df_sim["caseid"].nunique()

    if verbose:
        sep = "-" * 65
        print(f"\n{sep}")
        print(f"  EVALUACION DE METRICAS BPS (Graziosi et al. 2024)")
        print(f"{sep}")
        print(f"  Referencia : {Path(ref_path).name}"
              f"  ({n_cases_ref} casos, {len(df_ref)} eventos)")
        print(f"  Simulado   : {Path(sim_path).name}"
              f"  ({n_cases_sim} casos, {len(df_sim)} eventos)")
        print(f"{sep}\n")

    results: Dict = {}

    # --- RED ---
    try:
        red = compute_red(df_ref, df_sim)
        results["RED"] = red
        if verbose:
            print(f"  RED   Relative Event Distribution  : {red:.6f}"
                  f"  [0=identico, 1=max diferencia, EMD posicion relativa en traza]")
    except Exception as exc:
        results["RED"] = None
        if verbose:
            print(f"  RED   Relative Event Distribution  : N/A  ({exc})")

    # --- CTD ---
    try:
        ctd = compute_ctd(df_ref, df_sim)
        ctd_h = ctd / 3600
        results["CTD"]       = ctd
        results["CTD_horas"] = ctd_h
        if verbose:
            print(f"  CTD   Cycle Time Distribution      : {ctd:,.0f} s"
                  f"  ({ctd_h:.2f} h)  [EMD tiempos de ciclo]")
    except Exception as exc:
        results["CTD"] = results["CTD_horas"] = None
        if verbose:
            print(f"  CTD   Cycle Time Distribution      : N/A  ({exc})")

    # --- 2GD ---
    try:
        gd2 = compute_2gd(df_ref, df_sim)
        results["2GD"] = gd2
        if verbose:
            print(f"  2GD   2-Gram Distance              : {gd2:.6f}"
                  f"  [0 = identico, 1 = max diferencia, EMD bigramas]")
    except Exception as exc:
        results["2GD"] = None
        if verbose:
            print(f"  2GD   2-Gram Distance              : N/A  ({exc})")

    # --- CONF ---
    try:
        conf, constraint_strings = compute_conf(
            df_ref, df_sim,
            support_threshold=support_threshold,
            cached_model=cached_model,
        )
        n_rules = len(constraint_strings)
        results["CONF"]    = conf
        results["n_rules"] = n_rules
        if verbose:
            print(f"  CONF  Conformance Score            : {conf:.2%}"
                  f"  [{n_rules} restricciones DECLARE (MINERful), soporte>={support_threshold:.0%}]")
    except Exception as exc:
        results["CONF"] = results["n_rules"] = None
        if verbose:
            print(f"  CONF  Conformance Score            : N/A  ({exc})")

    if verbose:
        print(f"\n{'-' * 65}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluacion en lote
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    ref_path: str,
    sim_paths: List[str],
    support_threshold: float = 0.90,
    cached_model: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Evalua multiples logs simulados contra el mismo log de referencia.

    MINERful se ejecuta UNA SOLA VEZ sobre el log de referencia para
    descubrir las restricciones DECLARE; el modelo resultante se reutiliza
    en todos los logs simulados (eficiencia en batch).

    Retorna DataFrame con columnas: simulado, RED, CTD, CTD_horas, 2GD, CONF, n_rules.
    """
    # Ejecutar MINERful una sola vez si no se provee modelo cacheado
    if cached_model is None:
        df_ref = load_log(ref_path)
        cached_model, constraint_strings = discover_conf_model(df_ref, support_threshold)
        if cached_model:
            n = len(constraint_strings)
            print(f"[CONF] MINERful: {n} restricciones DECLARE"
                  f" (soporte >= {support_threshold:.0%})")
        else:
            print(f"[CONF] Advertencia: MINERful no genero restricciones.")

    rows = []
    for sim_path in sim_paths:
        r = evaluate(
            ref_path=ref_path,
            sim_path=sim_path,
            support_threshold=support_threshold,
            cached_model=cached_model,
            verbose=False,
        )
        r["simulado"] = Path(sim_path).parent.name
        rows.append(r)

    if not rows:
        return pd.DataFrame()

    cols = ["simulado", "RED", "CTD", "CTD_horas", "2GD", "CONF", "n_rules"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluator",
        description="Evalua metricas de calidad BPS: RED, CTD, 2GD, CONF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "reference",
        help="CSV del log de referencia (ASIS preprocesado)",
    )
    parser.add_argument(
        "simulated",
        nargs="+",
        help="Uno o mas CSV de logs simulados (output Prosimos TOBE)",
    )
    parser.add_argument(
        "--support",
        dest="support_threshold",
        type=float,
        default=0.90,
        metavar="UMBRAL",
        help="Umbral de soporte para mineria DECLARE (default: 0.90)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="resultado.csv",
        help="(Opcional) Guardar tabla resumen en CSV",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if len(args.simulated) == 1:
        evaluate(
            ref_path=args.reference,
            sim_path=args.simulated[0],
            support_threshold=args.support_threshold,
        )
    else:
        df = evaluate_batch(
            ref_path=args.reference,
            sim_paths=args.simulated,
            support_threshold=args.support_threshold,
        )
        print(df.to_string(index=False))

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nTabla guardada en: {args.output}")


if __name__ == "__main__":
    main()
