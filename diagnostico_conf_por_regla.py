# -*- coding: utf-8 -*-
"""
Diagnostico fino de CONF: compara, regla por regla, el 'Trace support' que
reporto MINERful contra el que implicitamente mide pm4py al verificar las
mismas trazas contra la misma regla.

Por que este diagnostico
-------------------------
El self-check (verificar_conf_selfcheck.py) probo que comparar test_split.csv
contra si mismo da CONF=48.78%, cuando matematicamente deberia dar >=90%
(las 41 reglas se filtraron exigiendo Trace support >= 90% sobre ese mismo
log, y dev_fitness de pm4py es fraccional: 1 - deviaciones/total_restricciones
-- promediar 41 numeros que son todos >=90% no puede dar 48.78%).

Eso confirma que hay una regla (o varias) donde MINERful y pm4py estan en
desacuerdo sobre que trazas la cumplen. Este script aisla exactamente cual.

Ejecutar desde la raiz del proyecto, en el venv deep_generator:
    python diagnostico_conf_por_regla.py
"""
import sys
from pathlib import Path

TEST_SPLIT = Path("data/4.simulation_results/RunningExample/metricas/test_split.csv")
SUPPORT_THRESHOLD = 0.90


def main():
    if not TEST_SPLIT.exists():
        print(f"[ERROR] No se encontro: {TEST_SPLIT}")
        sys.exit(1)

    from evaluacion.metrics import (
        load_log, _run_minerful, _minerful_to_pm4py_model,
    )
    from pm4py.algo.conformance.declare.variants.classic import apply_list

    df_ref = load_log(str(TEST_SPLIT))
    n_traces = df_ref["caseid"].nunique()
    print(f"Log: {TEST_SPLIT}  ({n_traces} casos)\n")

    print("Ejecutando MINERful...")
    rules_df = _run_minerful(df_ref, SUPPORT_THRESHOLD)
    if rules_df is None or rules_df.empty:
        print("[ERROR] MINERful no genero reglas.")
        sys.exit(1)

    model = _minerful_to_pm4py_model(rules_df, support_threshold=SUPPORT_THRESHOLD)
    if not model:
        print("[ERROR] El modelo pm4py quedo vacio.")
        sys.exit(1)

    # ── Reconstruir, para cada constraint del modelo, el Trace support que ──
    # ── reporto MINERful (misma logica de _minerful_to_pm4py_model) ─────────
    cols_lower = {c.lower().strip(): c for c in rules_df.columns}
    constraint_col = cols_lower.get("constraint")
    trace_sup_col  = cols_lower.get("trace support")

    def _clean(raw):
        s = str(raw).strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        return s.strip().strip("'\"").strip()

    from evaluacion.metrics import _MINERFUL_TO_PM4PY
    import re as _re

    minerful_sup = {}   # (tmpl_lower, key) -> trace_sup reportado por MINERful
    activation_col = cols_lower.get("activation")
    target_col     = cols_lower.get("target")
    for _, row in rules_df.iterrows():
        raw = str(row[constraint_col]).strip("'").strip()
        m = _re.match(r"(\w+)\((.+)\)", raw)
        if not m:
            continue
        tmpl = _MINERFUL_TO_PM4PY.get(m.group(1).lower())
        if tmpl is None:
            continue
        sup = float(row[trace_sup_col]) if trace_sup_col else None
        if activation_col and target_col:
            act = _clean(row[activation_col])
            tgt = _clean(row[target_col])
            key = tgt if not act else (act, tgt)
        else:
            args = [a.strip() for a in m.group(2).split(",")]
            key = args[0] if len(args) == 1 else (args[0], args[1])
        minerful_sup[(tmpl.lower(), key)] = sup

    # ── Proyectar el log y correr pm4py ──────────────────────────────────────
    ts_col = "start_timestamp" if "start_timestamp" in df_ref.columns else "end_timestamp"
    projected_log = []
    for _, grp in df_ref.sort_values(["caseid", ts_col]).groupby("caseid", sort=False):
        projected_log.append(grp.sort_values(ts_col)["task"].tolist())

    print("Ejecutando pm4py.apply_list...\n")
    results = apply_list(projected_log, model)

    # ── Tabular violaciones por constraint ───────────────────────────────────
    from collections import Counter
    violations = Counter()
    for r in results:
        for dev in r.get("deviations", []):
            tmpl_raw, key = dev[0], dev[1]
            violations[(str(tmpl_raw).lower(), key)] += 1

    # ── Comparar pm4py (implicito) vs MINERful, por constraint ──────────────
    rows = []
    for tmpl, cdict in model.items():
        for key in cdict:
            k = (tmpl.lower(), key)
            n_viol = violations.get(k, 0)
            pm4py_sup = 1.0 - (n_viol / n_traces)
            mf_sup = minerful_sup.get(k)
            gap = (mf_sup - pm4py_sup) if mf_sup is not None else float('nan')
            rows.append((tmpl, key, mf_sup, pm4py_sup, n_viol, gap))

    rows.sort(key=lambda r: (r[5] if r[5] == r[5] else -1), reverse=True)  # nan-safe desc

    print(f"{'Plantilla':<16}{'Argumentos':<45}{'MINERful':>10}{'pm4py':>10}{'#Viol':>8}{'Gap':>8}")
    print("-" * 100)
    for tmpl, key, mf_sup, pm4py_sup, n_viol, gap in rows:
        key_str = str(key)
        if len(key_str) > 43:
            key_str = key_str[:40] + "..."
        mf_str = f"{mf_sup:.1%}" if mf_sup is not None else "N/A"
        print(f"{tmpl:<16}{key_str:<45}{mf_str:>10}{pm4py_sup:>10.1%}{n_viol:>8}{gap:>8.1%}"
              if gap == gap else
              f"{tmpl:<16}{key_str:<45}{mf_str:>10}{pm4py_sup:>10.1%}{n_viol:>8}{'N/A':>8}")

    n_big_gap = sum(1 for r in rows if r[5] == r[5] and r[5] > 0.05)
    print("\n" + "=" * 100)
    print(f"Reglas con gap > 5 puntos porcentuales entre MINERful y pm4py: "
          f"{n_big_gap} / {len(rows)}")
    print("Estas son las reglas donde los dos sistemas estan en desacuerdo "
          "sobre que trazas la cumplen. Revisa las de mayor gap primero.")
    print("=" * 100)


if __name__ == "__main__":
    main()
