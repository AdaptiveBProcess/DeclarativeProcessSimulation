"""
Calculo de uplift y ranking por score de Welch (CLAUDE.md Seccion 12.3-12.4).

Para cada regla candidata: separa las trazas en cumple/no-cumple, calcula el uplift
de PCE entre ambos grupos, y castiga la incertidumbre con el limite inferior de un
intervalo de confianza al 95% (usando la t de Welch-Satterthwaite en vez de un
z=1.96 fijo, para no necesitar un `n_minimo` arbitrario -- ver CLAUDE.md 12.4).

    SE(c)    = sqrt(var_cumple/n_cumple + var_no_cumple/n_no_cumple)
    df(c)    = Welch-Satterthwaite, a partir de var/n de cada grupo
    score(c) = uplift(c) - t_{0.975; df(c)} * SE(c)

Piso matematico (no un parametro a ajustar): una candidata solo es "scoreable" si
ambos grupos tienen al menos 2 trazas -- si no, la varianza no esta definida.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import t as student_t

from rule_selection.trace_index import TraceIndex


def compute_compliance_table(
    trace_indexes: dict[str, TraceIndex], df_candidates: pd.DataFrame
) -> pd.DataFrame:
    """
    Matriz traza x candidata (booleana), evaluando cada candidata contra cada traza
    usando los TraceIndex ya precomputados (una sola vez por traza).

    `df_candidates` debe tener las columnas `candidate_id`, `rule`, `arg0`, y
    opcionalmente `arg1` (NaN/None para reglas unarias: required, not_allowed).
    """
    trace_ids = list(trace_indexes.keys())
    data: dict[str, list[bool]] = {}

    has_arg1 = "arg1" in df_candidates.columns

    for _, row in df_candidates.iterrows():
        candidate_id = row["candidate_id"]
        rule = row["rule"]
        arg0 = row["arg0"]
        arg1 = row["arg1"] if has_arg1 and pd.notna(row["arg1"]) else None

        data[candidate_id] = [
            trace_indexes[trace_id].evaluate(rule, arg0, arg1) for trace_id in trace_ids
        ]

    return pd.DataFrame(data, index=trace_ids)


def compute_uplift_stats(compliance_wide: pd.DataFrame, pce_by_trace: pd.Series) -> pd.DataFrame:
    """
    Por cada candidata (columna de `compliance_wide`): n_cumple, n_no_cumple, medias
    de PCE, uplift, varianzas (ddof=1), SE, df de Welch-Satterthwaite, t critico al
    97.5%, score, y si la candidata es evaluable (`scoreable`).

    `pce_by_trace` debe estar indexado por el mismo identificador de traza que las
    filas de `compliance_wide` (ej. la columna `traza`/`PCE` de
    `generar_analisis_por_traza`, ya indexada por `traza`).

    Los identificadores de traza se comparan como string en ambos lados antes de
    alinear, sin importar el dtype real de cada indice -- `trace_index.build_trace_sequences`
    usa `str(case_id)` como clave, pero `case:concept:name` (y por lo tanto la
    columna `traza` de `generar_analisis_por_traza`) suele quedar como `Int64` tras
    `convertir_int`. Sin esta normalizacion, `reindex` compara `'265' == 265` como
    `False` y el resultado queda en NaN para TODAS las candidatas -- bug real
    encontrado corriendo esto contra RunningExample.csv (19 candidatas con splits
    validos de cumplimiento, las 19 con uplift=NaN antes de este fix).
    """
    pce_aligned = pce_by_trace.copy()
    pce_aligned.index = pce_aligned.index.astype(str)
    pce_aligned = pce_aligned.reindex(compliance_wide.index.astype(str)).astype(float)

    rows = []
    for candidate_id in compliance_wide.columns:
        cumple_mask = compliance_wide[candidate_id].astype(bool)
        pce_cumple = pce_aligned[cumple_mask]
        pce_no_cumple = pce_aligned[~cumple_mask]

        n_cumple = int(pce_cumple.shape[0])
        n_no_cumple = int(pce_no_cumple.shape[0])

        mean_cumple = float(pce_cumple.mean()) if n_cumple > 0 else float("nan")
        mean_no_cumple = float(pce_no_cumple.mean()) if n_no_cumple > 0 else float("nan")
        uplift = (
            mean_cumple - mean_no_cumple
            if n_cumple > 0 and n_no_cumple > 0
            else float("nan")
        )

        var_cumple = float(pce_cumple.var(ddof=1)) if n_cumple >= 2 else float("nan")
        var_no_cumple = float(pce_no_cumple.var(ddof=1)) if n_no_cumple >= 2 else float("nan")

        se = t_crit = score = welch_df = float("nan")
        scoreable = n_cumple >= 2 and n_no_cumple >= 2

        if scoreable:
            term_cumple = var_cumple / n_cumple
            term_no_cumple = var_no_cumple / n_no_cumple
            se = math.sqrt(term_cumple + term_no_cumple)

            numerator = (term_cumple + term_no_cumple) ** 2
            denominator = (term_cumple**2) / (n_cumple - 1) + (term_no_cumple**2) / (
                n_no_cumple - 1
            )

            if denominator > 0:
                welch_df = numerator / denominator
                t_crit = float(student_t.ppf(0.975, welch_df))
                score = uplift - t_crit * se
            else:
                # Ambos grupos con varianza 0 -- no hay dispersion que respalde un
                # intervalo de confianza. Se descarta en vez de forzar un score.
                scoreable = False

        rows.append(
            {
                "candidate_id": candidate_id,
                "n_cumple": n_cumple,
                "n_no_cumple": n_no_cumple,
                "mean_pce_cumple": mean_cumple,
                "mean_pce_no_cumple": mean_no_cumple,
                "uplift": uplift,
                "var_cumple": var_cumple,
                "var_no_cumple": var_no_cumple,
                "SE": se,
                "df": welch_df,
                "t_crit": t_crit,
                "score": score,
                "scoreable": scoreable,
            }
        )

    return pd.DataFrame(rows)


def rank_candidates(stats_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Filtra las candidatas evaluables y devuelve las `top_n` de mayor score.

    Desempate por `candidate_id` (orden alfabetico) cuando el score empata
    exactamente -- encontrado en la practica contra RunningExample.csv real: 9
    candidatas relacionadas con `Task C` comparten `n_cumple`/`n_no_cumple`/uplift
    identicos (su actividad "objetivo" tambien esta en las 540 trazas, asi que el
    split de cumplimiento coincide), y sin un desempate explicito, cual de ellas
    entraba al top-5 dependia del orden de llegada de los datos -- no determinista,
    justo lo que el rediseno de CLAUDE.md 12 buscaba eliminar.
    """
    scoreable = stats_df[stats_df["scoreable"]].copy()
    return (
        scoreable.sort_values(["score", "candidate_id"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
