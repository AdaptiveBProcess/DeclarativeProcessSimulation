"""
Valida rule_selection.uplift contra el ejemplo trabajado de CLAUDE.md Seccion 12.4.1
(12 trazas de juguete, 3 candidatas: Precedence(B,D), Coexistence(C,F), Required(A)).
"""

import pandas as pd
import pytest

from rule_selection.trace_index import build_trace_indexes
from rule_selection.uplift import compute_compliance_table, compute_uplift_stats, rank_candidates
from rule_selection.tests.fixtures.toy_log import TOY_SEQUENCES, TOY_PCE


CANDIDATES = pd.DataFrame(
    [
        {"candidate_id": "precedence_B_D", "rule": "precedence", "arg0": "B", "arg1": "D"},
        {"candidate_id": "coexistence_C_F", "rule": "coexistence", "arg0": "C", "arg1": "F"},
        {"candidate_id": "required_A", "rule": "required", "arg0": "A", "arg1": None},
    ]
)


@pytest.fixture
def stats_df():
    indexes = build_trace_indexes(TOY_SEQUENCES)
    compliance = compute_compliance_table(indexes, CANDIDATES)
    pce = pd.Series(TOY_PCE)
    return compute_uplift_stats(compliance, pce).set_index("candidate_id")


def test_precedence_b_d_worked_example(stats_df):
    row = stats_df.loc["precedence_B_D"]

    assert row["n_cumple"] == 7
    assert row["n_no_cumple"] == 5
    assert row["uplift"] == pytest.approx(32.23, abs=0.05)
    assert row["var_cumple"] == pytest.approx(352.62, abs=0.05)
    assert row["var_no_cumple"] == pytest.approx(635.70, abs=0.05)
    assert row["SE"] == pytest.approx(13.32, abs=0.05)
    assert row["df"] == pytest.approx(7.06, abs=0.05)
    assert row["t_crit"] == pytest.approx(2.36, abs=0.05)
    assert row["score"] == pytest.approx(0.78, abs=0.1)
    assert row["scoreable"] is True or row["scoreable"] == True  # noqa: E712 (numpy bool)


def test_coexistence_c_f_excluded_by_floor(stats_df):
    row = stats_df.loc["coexistence_C_F"]
    assert row["n_cumple"] == 11
    assert row["n_no_cumple"] == 1
    assert not row["scoreable"]


def test_required_a_excluded_by_floor(stats_df):
    row = stats_df.loc["required_A"]
    assert row["n_cumple"] == 12
    assert row["n_no_cumple"] == 0
    assert not row["scoreable"]


def test_rank_candidates_keeps_only_scoreable(stats_df):
    ranked = rank_candidates(stats_df.reset_index(), top_n=5)
    assert list(ranked["candidate_id"]) == ["precedence_B_D"]


def test_compute_uplift_stats_handles_mismatched_index_dtypes():
    """
    Test de regresion: encontrado corriendo el pipeline contra RunningExample.csv
    real. `trace_index.build_trace_sequences` usa `str(case_id)` como identificador
    de traza, pero `case:concept:name` (y por lo tanto la columna `traza` de
    `generar_analisis_por_traza`) suele llegar como enteros (`Int64`, tras
    `convertir_int`). Sin normalizar ambos lados a string antes de alinear,
    `pce_by_trace.reindex(...)` no encuentra ningun match ('265' != 265) y el
    uplift de TODAS las candidatas queda en NaN -- silenciosamente, sin error.
    """
    sequences = {"0": ["A", "B"], "1": ["A"], "2": ["A", "B"], "3": ["A"]}
    indexes = build_trace_indexes(sequences)

    candidates = pd.DataFrame([{"candidate_id": "required_B", "rule": "required", "arg0": "B", "arg1": None}])
    compliance = compute_compliance_table(indexes, candidates)

    # pce_by_trace con indice ENTERO (no string) -- asi llega en la practica desde
    # generar_analisis_por_traza tras convertir_int sobre case:concept:name.
    pce_by_trace = pd.Series({0: 80.0, 1: 20.0, 2: 90.0, 3: 10.0})
    assert pce_by_trace.index.dtype != compliance.index.dtype  # confirma que el caso de prueba es real

    stats = compute_uplift_stats(compliance, pce_by_trace).set_index("candidate_id")
    row = stats.loc["required_B"]

    assert row["n_cumple"] == 2  # trazas 0 y 2 (las que tienen B)
    assert row["n_no_cumple"] == 2  # trazas 1 y 3
    assert not pd.isna(row["uplift"]), "uplift no deberia quedar en NaN por un mismatch de dtype de indice"
    assert row["mean_pce_cumple"] == pytest.approx(85.0)  # media de 80 y 90
    assert row["mean_pce_no_cumple"] == pytest.approx(15.0)  # media de 20 y 10
    assert row["uplift"] == pytest.approx(70.0)


def test_rank_candidates_breaks_ties_deterministically_by_candidate_id():
    """
    Test de regresion: encontrado corriendo el pipeline contra RunningExample.csv
    real -- 9 candidatas empataban exactamente en score, y sin desempate explicito
    el top-5 cambiaba entre corridas segun el orden de llegada de las filas.
    """
    base_row = {
        "n_cumple": 265,
        "n_no_cumple": 275,
        "mean_pce_cumple": 108.13,
        "mean_pce_no_cumple": 100.0,
        "uplift": 8.13,
        "var_cumple": 149.85,
        "var_no_cumple": 0.0,
        "SE": 0.75,
        "df": 264.0,
        "t_crit": 1.97,
        "score": 6.65,  # el mismo score, a proposito, para las 3 primeras
        "scoreable": True,
    }
    stats_df = pd.DataFrame(
        [
            {**base_row, "candidate_id": "zzz_last_alphabetically"},
            {**base_row, "candidate_id": "aaa_first_alphabetically"},
            {**base_row, "candidate_id": "mmm_middle"},
            {**base_row, "candidate_id": "clear_winner", "score": 12.23, "uplift": 14.37},
        ]
    )

    ranked_once = rank_candidates(stats_df, top_n=3)
    # shuffle de las filas -- mismo contenido, distinto orden de llegada
    ranked_twice = rank_candidates(stats_df.sample(frac=1, random_state=42), top_n=3)

    assert list(ranked_once["candidate_id"]) == list(ranked_twice["candidate_id"])
    assert list(ranked_once["candidate_id"]) == [
        "clear_winner",
        "aaa_first_alphabetically",
        "mmm_middle",
    ]
