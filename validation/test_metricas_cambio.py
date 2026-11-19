"""Unit tests for evaluacion/metricas_cambio.py (quality metrics under a change phi).

Every expected value is small enough to be checked by hand.

Run from the repository root:
    python -m pytest validation/test_metricas_cambio.py -v
"""
import math
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluacion import metricas_cambio as mc  # noqa: E402

C = {'support': 100, 'confidence': 100}


def make_log(traces, roles=None):
    """traces: list of activity lists (one case each). roles: optional parallel lists."""
    rows = []
    for i, tasks in enumerate(traces):
        for j, task in enumerate(tasks):
            row = {
                'caseid': f'c{i}',
                'task': task,
                'start_timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=10 * i + j),
                'end_timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=10 * i + j, minutes=30),
            }
            if roles is not None:
                row['role'] = roles[i][j]
            rows.append(row)
    return pd.DataFrame(rows)


TRAIN = make_log([list('ABC'), list('ACB')])
SIM = make_log([list('ABC'), list('ABD'), list('ABD')])
PHI_A_D = {'rule': 'directly', 'path': ['A', 'D']}      # D right after A, at least once
PHI_REQ_D = {'rule': 'required', 'path': ['D']}


# ── Novelty ──────────────────────────────────────────────────────────────────

def test_novedad_por_traza_y_por_variante():
    out = mc.compute_novedad(TRAIN, SIM)
    assert out['novedad_trazas'] == pytest.approx(2 / 3)   # ABD, ABD are new
    assert out['novedad_variantes'] == pytest.approx(1 / 2)  # {ABC, ABD}: only ABD new


def test_novedad_no_pedida_ignora_actividades_de_phi():
    # Without A and D: train -> {BC, CB}; sim -> BC (seen), B (new), B (new)
    assert mc.compute_novedad_no_pedida(TRAIN, SIM, PHI_A_D) == pytest.approx(2 / 3)


def test_novedad_no_pedida_es_cero_si_phi_explica_todo_lo_nuevo():
    sim = make_log([list('ADBC')])          # ADBC is new, but without A/D it is BC
    assert mc.compute_novedad_no_pedida(TRAIN, sim, PHI_A_D) == 0.0


def test_novedad_validada_contra_referencia():
    ref = make_log([list('ABD')])
    out = mc.compute_novedad_validada(TRAIN, SIM, ref)
    assert out['n_nuevas'] == 2
    assert out['novedad_validada'] == 1.0


# ── Conditional control ──────────────────────────────────────────────────────

def test_control_proporcion_y_error():
    out = mc.compute_control(SIM, PHI_REQ_D, p_pedida=0.5)
    assert out['prop_cumple'] == pytest.approx(2 / 3)
    assert out['error_proporcion'] == pytest.approx(2 / 3 - 0.5)


def test_control_por_clase_pedida():
    sim = SIM.copy()
    pedido = {'c0': 'si', 'c1': 'si', 'c2': 'no'}   # c0 asked to comply but does not
    sim['label'] = sim['caseid'].map(pedido)
    out = mc.compute_control(sim, PHI_REQ_D, col_pedido='label', valor_cumplir='si')
    assert out['tasa_cumple_pedidas'] == pytest.approx(1 / 2)   # c1 yes, c0 no
    assert out['tasa_evita_no_pedidas'] == 0.0                  # c2 complies anyway


# ── Plausibility conditioned on phi ──────────────────────────────────────────

MODELO = {
    'chainresponse': {('A', 'B'): C},   # every A immediately followed by B -> clashes with A>>D
    'response': {('A', 'C'): C},        # A ... C: compatible with A>>D (A D C)
    'absence': {'D': C},                # D never: clashes with A>>D
    'existence': {'B': C},              # does not touch phi's activities
}


def test_restricciones_contradichas_por_phi():
    contradichas = mc.restricciones_contradichas(MODELO, PHI_A_D)
    assert contradichas == {('chainresponse', ('A', 'B')), ('absence', 'D')}


def test_filtrar_modelo_quita_solo_las_contradichas():
    filtrado = mc.filtrar_modelo(MODELO, {('absence', 'D')})
    assert 'absence' not in filtrado
    assert ('A', 'B') in filtrado['chainresponse']


def test_alcance_de_cada_restriccion():
    assert mc.alcance_restriccion(('A', 'D'), PHI_A_D) == 'local'
    assert mc.alcance_restriccion(('A', 'C'), PHI_A_D) == 'vecindario'
    assert mc.alcance_restriccion('B', PHI_A_D) == 'global'


def test_plausibilidad_media_estricta_y_por_alcance():
    filtrado = mc.filtrar_modelo(MODELO, mc.restricciones_contradichas(MODELO, PHI_A_D))
    # Remaining: response(A, C) [vecindario], existence(B) [global]
    sim = make_log([list('ADBC'), list('ADB'), list('ADC')])
    out = mc.compute_plausibilidad(sim, filtrado, PHI_A_D)
    # ADBC: 2/2 ok; ADB: response violated (1/2); ADC: existence(B) violated (1/2)
    assert out['plaus_media'] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert out['plaus_estricta'] == pytest.approx(1 / 3)
    assert out['plaus_estricta_vecindario'] == pytest.approx(2 / 3)
    assert out['plaus_estricta_global'] == pytest.approx(2 / 3)
    assert math.isnan(out['plaus_estricta_local'])


def test_tasa_violacion_por_restriccion():
    ocultas = make_log([list('ADC'), list('ADBC')])
    tabla = mc.tasa_violacion_por_restriccion(ocultas, MODELO).set_index('plantilla')
    assert tabla.loc['absence', 'tasa_violacion'] == 1.0         # both contain D
    assert tabla.loc['chainresponse', 'tasa_violacion'] == 1.0   # A is followed by D
    assert tabla.loc['response', 'tasa_violacion'] == 0.0


def test_compatibilidad_rol_actividad():
    train = make_log([list('AB')], roles=[['r1', 'r2']])
    sim = make_log([list('AB')], roles=[['r1', 'r1']])           # (r1, B) never seen
    assert mc.compute_compatibilidad_rol_actividad(train, sim) == pytest.approx(1 / 2)
    assert math.isnan(mc.compute_compatibilidad_rol_actividad(TRAIN, SIM))  # no role column


# ── Fidelity ─────────────────────────────────────────────────────────────────

def test_freq_actividades_tvd():
    ref = make_log([list('AB')])          # A 1/2, B 1/2
    sim = make_log([list('AA')])          # A 1
    assert mc.compute_freq_actividades(ref, sim) == pytest.approx(0.5)
    assert mc.compute_freq_actividades(ref, ref) == 0.0


def test_fidelidad_no_tocada_ignora_actividades_de_phi():
    ref = make_log([list('BC')])
    sim = make_log([list('ADBC')])        # without A and D it is exactly BC
    out = mc.compute_fidelidad_no_tocada(ref, sim, PHI_A_D)
    assert out['2GD_no_tocada'] == 0.0
    assert out['freq_act_no_tocada'] == 0.0


# ── Diversity ────────────────────────────────────────────────────────────────

def test_damerau_levenshtein_cuenta_un_intercambio_como_una_operacion():
    assert mc._damerau_levenshtein('ABCD', 'ACBD') == 1
    assert mc._damerau_levenshtein('ABCD', 'ABCE') == 1
    assert mc._damerau_levenshtein('ACBD', 'ABCE') == 2
    assert mc._damerau_levenshtein('', 'AB') == 2


def test_diversidad_ejemplo_trabajado():
    log = make_log([list('ABCD'), list('ACBD'), list('ABCE')])
    out = mc.compute_diversidad(log)
    assert out['diversidad_dl'] == pytest.approx((0.25 + 0.25 + 0.5) / 3)
    assert out['razon_variantes'] == 1.0


def test_diversidad_muestrea_cuando_hay_demasiados_pares():
    log = make_log([list('AB'), list('AB'), list('AC'), list('AC')])
    out = mc.compute_diversidad(log, max_pares=3, semilla=1)
    assert 0.0 <= out['diversidad_dl'] <= 1.0
    assert out['razon_variantes'] == pytest.approx(0.5)


# ── Valid novelty and full evaluation ────────────────────────────────────────

def test_novedad_valida():
    filtrado = mc.filtrar_modelo(MODELO, mc.restricciones_contradichas(MODELO, PHI_A_D))
    sim = make_log([list('ADBC'), list('ADB'), list('ABC')])
    out = mc.compute_novedad_valida(TRAIN, sim, PHI_A_D, filtrado)
    # ADBC: new, phi, plausible -> valid; ADB: violates response(A, C); ABC: seen, no phi
    assert out['novedad_valida'] == pytest.approx(1 / 3)
    assert out['novedad_valida_entre_phi'] == pytest.approx(1 / 2)


def test_evaluar_cambio_devuelve_todas_las_metricas():
    sim = make_log([list('ADBC'), list('ADB'), list('ABC')])
    out = mc.evaluar_cambio(TRAIN, sim, PHI_A_D, MODELO, df_ref_novedad=sim, p_pedida=0.5)
    for clave in ('n_contradichas', 'novedad_trazas', 'novedad_no_pedida', 'novedad_validada',
                  'prop_cumple', 'error_proporcion', 'plaus_media', 'plaus_estricta',
                  'novedad_valida', 'freq_act', '2GD_no_tocada', 'diversidad_dl',
                  'diversidad_dl_ref'):
        assert clave in out
    assert out['n_contradichas'] == 2
