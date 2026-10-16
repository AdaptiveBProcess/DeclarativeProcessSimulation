"""Unit tests for the DECLARE templates evaluated in support_modules/traces_evaluation.py.

Run from the repository root:
    python -m pytest validation/test_plantillas_declare.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support_modules import traces_evaluation as te  # noqa: E402

ACTIVITIES = ['A', 'B', 'C', 'D', 'E']
AC_INDEX = {a: i for i, a in enumerate(ACTIVITIES)}


def make_case(tasks, caseid='c1'):
    return pd.DataFrame({
        'caseid': [caseid] * len(tasks),
        'task': tasks,
        'start_timestamp': pd.date_range('2024-01-01', periods=len(tasks), freq='h'),
    })


def write_ini(tmp_path, body):
    path = tmp_path / 'rules.ini'
    path.write_text('[RULES]\n' + body + '\n', encoding='utf-8')
    return str(path)


# --- extract_rules ---------------------------------------------------------

def test_parse_any_of(tmp_path):
    rules = te.extract_rules(write_ini(tmp_path, 'any_of = Appeal to Judge ; Send Appeal ; C'))
    assert rules['rule'] == 'any_of'
    assert rules['path'] == ['Appeal to Judge', 'Send Appeal', 'C']


def test_parse_min_occurrences(tmp_path):
    rules = te.extract_rules(write_ini(tmp_path, 'path = O_CREATED-COMPLETE\nmin_occurrences = 2'))
    assert rules['rule'] == 'atleast_2'
    assert rules['path'] == ['O_CREATED-COMPLETE']


def test_parse_existing_templates_unchanged(tmp_path):
    assert te.extract_rules(write_ini(tmp_path, 'path = O_DECLINED-COMPLETE'))['rule'] == 'required'
    rules = te.extract_rules(write_ini(tmp_path, 'path = EVENT 1 START >> + >> Task D\nvariation = =1'))
    assert rules['rule'] == 'coexistence'
    assert rules['path'] == ['EVENT 1 START', 'Task D']
    assert te.extract_rules(write_ini(tmp_path, 'path = A >> C'))['rule'] == 'directly'


def test_parse_min_occurrences_rejects_invalid(tmp_path):
    with pytest.raises(ValueError):
        te.extract_rules(write_ini(tmp_path, 'path = A >> B\nmin_occurrences = 2'))
    with pytest.raises(ValueError):
        te.extract_rules(write_ini(tmp_path, 'path = A\nmin_occurrences = 0'))


# --- evaluate_condition (DataFrame, used by the GAN) -----------------------

@pytest.mark.parametrize('tasks, expected', [
    (['A', 'B', 'B', 'E'], True),
    (['A', 'B', 'E'], False),
    (['A', 'E'], False),
])
def test_atleast_2(tasks, expected):
    assert te.evaluate_condition(make_case(tasks), AC_INDEX, ['B'], 'atleast_2') is expected


@pytest.mark.parametrize('tasks, expected', [
    (['A', 'C', 'E'], True),
    (['A', 'D', 'E'], True),
    (['A', 'B', 'E'], False),
])
def test_any_of(tasks, expected):
    assert te.evaluate_condition(make_case(tasks), AC_INDEX, ['C', 'D'], 'any_of') is expected


def test_any_of_with_activity_outside_vocabulary():
    assert te.evaluate_condition(make_case(['A', 'C']), AC_INDEX, ['ZZZ', 'C'], 'any_of') is True
    assert te.evaluate_condition(make_case(['A', 'B']), AC_INDEX, ['ZZZ', 'C'], 'any_of') is False


# --- evaluate_condition_list (step-wise check used by the LSTM) -------------

def test_list_version_matches():
    seq = [AC_INDEX[a] for a in ['A', 'B', 'B']]
    assert te.evaluate_condition_list(seq, AC_INDEX, ['B'], 'atleast_2') is True
    assert te.evaluate_condition_list(seq[:2], AC_INDEX, ['B'], 'atleast_2') is False
    assert te.evaluate_condition_list(seq, AC_INDEX, ['C', 'B'], 'any_of') is True
    assert te.evaluate_condition_list(seq, AC_INDEX, ['C', 'D'], 'any_of') is False


# --- case_satisfies_rule (used by the trainers for the training proportion) --

@pytest.mark.parametrize('rule, act_paths, tasks', [
    ('directly', ['A', 'C'], ['A', 'B', 'C', 'E']),
    ('directly', ['A', 'C'], ['A', 'C', 'E']),
    ('eventually', ['C', 'A'], ['A', 'B', 'C', 'E']),
    ('precedence', ['B', 'C'], ['A', 'C', 'B', 'E']),
    ('required', ['D'], ['A', 'D', 'E']),
    ('not_allowed', ['D'], ['A', 'D', 'E']),
    ('atleast_2', ['B'], ['A', 'B', 'B', 'E']),
    ('any_of', ['C', 'D'], ['A', 'B', 'E']),
])
def test_case_satisfies_rule_matches_evaluate_condition(rule, act_paths, tasks):
    case = make_case(tasks)
    expected = te.evaluate_condition(case.copy(), AC_INDEX, act_paths, rule)
    assert te.case_satisfies_rule(case, act_paths, rule) == expected


def test_trainer_bug_directly_is_not_coexistence():
    # The old trainer logic counted any trace containing A and C as compliant.
    assert te.case_satisfies_rule(make_case(['A', 'B', 'C', 'E']), ['A', 'C'], 'directly') is False
