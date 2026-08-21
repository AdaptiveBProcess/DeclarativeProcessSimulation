"""
Tests de rule_selection.trace_index.

1. Contra el ejemplo de juguete de CLAUDE.md Seccion 12.4.1 (oraculo escrito a mano).
2. Diferencial: TraceIndex.evaluate(...) debe coincidir siempre con
   support_modules.traces_evaluation.evaluate_condition_list(...) (sin modificar),
   para las 8 reglas, sobre secuencias aleatorias.
"""

import itertools
import random

import pytest

from rule_selection.trace_index import TraceIndex, build_trace_indexes, SUPPORTED_RULES
from rule_selection.tests.fixtures.toy_log import (
    TOY_SEQUENCES,
    PRECEDENCE_B_D_CUMPLE,
    PRECEDENCE_B_D_NO_CUMPLE,
)
from support_modules import traces_evaluation as te


def test_precedence_b_d_matches_worked_example():
    indexes = build_trace_indexes(TOY_SEQUENCES)

    cumple = {
        trace_id
        for trace_id, idx in indexes.items()
        if idx.evaluate("precedence", "B", "D")
    }
    no_cumple = set(TOY_SEQUENCES) - cumple

    assert cumple == PRECEDENCE_B_D_CUMPLE
    assert no_cumple == PRECEDENCE_B_D_NO_CUMPLE


def test_coexistence_c_f_matches_worked_example():
    # F esta en las 12 trazas; C esta en todas menos t9.
    indexes = build_trace_indexes(TOY_SEQUENCES)
    no_cumple = {t for t, idx in indexes.items() if not idx.evaluate("coexistence", "C", "F")}
    assert no_cumple == {"t9"}


def test_required_a_matches_worked_example():
    # A es la primera actividad de las 12 trazas, sin excepcion.
    indexes = build_trace_indexes(TOY_SEQUENCES)
    assert all(idx.evaluate("required", "A") for idx in indexes.values())


@pytest.mark.parametrize("seed", range(25))
def test_differential_vs_evaluate_condition_list(seed):
    """
    TraceIndex.evaluate nunca debe divergir de evaluate_condition_list (la funcion
    original, sin modificar) para ninguna de las 8 reglas.
    """
    rng = random.Random(seed)
    alphabet = ["A", "B", "C", "D", "E"]
    length = rng.randint(2, 8)
    seq = [rng.choice(alphabet) for _ in range(length)]

    # evaluate_condition_list construye el grafo G con los nombres de actividad tal
    # cual (u_tasks = set(list_case), sin pasar por ac_index), pero convierte
    # act_paths a traves de ac_index si la actividad esta ahi presente. Para que la
    # comparacion sea representacion-consistente (nombres de string en ambos lados,
    # igual que TraceIndex), se usa un ac_index vacio -- asi act_paths_idx nunca se
    # remapea a un entero y sigue comparando strings contra strings. Con un ac_index
    # no vacio, list_case tendria que venir codificado como enteros (asi lo llama en
    # produccion GenerativeLSTM/model_prediction/event_log_predictor.py, donde
    # seq_tasks + [x] son tokens enteros), que es un caso de uso distinto al de este
    # modulo (que trabaja con nombres de actividad de principio a fin).
    ac_index = {}
    idx = TraceIndex.from_sequence(seq)

    for rule in sorted(SUPPORTED_RULES):
        for arg0, arg1 in itertools.product(alphabet, alphabet):
            if rule in ("required", "not_allowed"):
                act_paths = [arg0]
                fast = idx.evaluate(rule, arg0)
            else:
                if arg0 == arg1:
                    continue  # las plantillas binarias de MINERful siempre tienen A != B
                act_paths = [arg0, arg1]
                fast = idx.evaluate(rule, arg0, arg1)

            slow = te.evaluate_condition_list(seq, ac_index, act_paths, rule)
            assert fast == slow, (
                f"Divergencia en seed={seed} seq={seq} rule={rule} "
                f"act_paths={act_paths}: fast={fast} slow={slow}"
            )
