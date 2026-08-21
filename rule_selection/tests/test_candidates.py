"""
Tests de rule_selection.candidates: filtrado a las 8 plantillas soportadas,
escritura de `.ini`, y compatibilidad con
`support_modules.traces_evaluation.extract_rules` (sin modificarla). No requieren
Java ni Docker -- trabajan sobre DataFrames sinteticos con el esquema real de
MINERful.
"""

import pandas as pd
import pytest

from rule_selection.candidates import (
    TEMPLATE_TO_RULE,
    DSL_FORMAT,
    parse_candidates,
    supported_candidates,
    write_rule_ini,
    top_support_candidate,
)
from support_modules.traces_evaluation import extract_rules


def _minerful_row(template, args, support=0.5, coverage=0.5, confidence=0.9):
    args_str = ", ".join(args)
    return {
        "Constraint": f"{template}({args_str})",
        "Template": template,
        "Activation": f"[{args[0]}]",
        "Target": f"[{args[1]}]" if len(args) > 1 else "",
        "Confidence": confidence,
        "Coverage": coverage,
        "Support": support,
    }


def _sample_raw_df():
    return pd.DataFrame(
        [
            _minerful_row("Absence", ["X"]),
            _minerful_row("Required", ["X"]),
            _minerful_row("Response", ["X", "Y"]),
            _minerful_row("AlternateResponse", ["X", "Y"]),
            _minerful_row("Precedence", ["X", "Y"]),
            _minerful_row("Succession", ["X", "Y"]),
            _minerful_row("CoExistence", ["X", "Y"]),
            _minerful_row("NotChainSuccession", ["X", "Y"]),
            # No soportadas por evaluate_condition/la alucinacion LSTM todavia --
            # deben conservarse en el pool (simulator_supported=False), no
            # descartarse (decision del usuario, 2026-08-10).
            _minerful_row("Init", ["X"]),
            _minerful_row("End", ["X"]),
            _minerful_row("Existence", ["X"]),
            _minerful_row("ChainResponse", ["X", "Y"]),
            _minerful_row("AtLeast3", ["X"]),
            _minerful_row("AtMost3", ["X"]),
        ]
    )


def test_parse_candidates_keeps_unsupported_templates_with_flag():
    df_parsed = parse_candidates(_sample_raw_df())

    # Las 14 filas sobreviven -- ninguna se descarta por plantilla.
    assert len(df_parsed) == 14

    supported = set(TEMPLATE_TO_RULE.keys())
    unsupported = {"init", "end", "existence", "chainresponse", "atleast3", "atmost3"}

    by_template = df_parsed.set_index("template")["simulator_supported"].to_dict()
    for template in supported:
        assert by_template[template] is True or by_template[template] == True  # noqa: E712
    for template in unsupported:
        assert by_template[template] is False or by_template[template] == False  # noqa: E712

    # Las no soportadas quedan con rule=None -- no hay como evaluarlas todavia.
    rule_for_unsupported = df_parsed.set_index("template").loc["atleast3", "rule"]
    assert pd.isna(rule_for_unsupported) or rule_for_unsupported is None


def test_parse_candidates_maps_rule_correctly_for_supported_templates():
    df_parsed = parse_candidates(_sample_raw_df())
    rule_by_template = (
        df_parsed[df_parsed["simulator_supported"]].set_index("template")["rule"].to_dict()
    )
    assert rule_by_template == TEMPLATE_TO_RULE


def test_supported_candidates_filters_down_to_the_8():
    df_parsed = parse_candidates(_sample_raw_df())
    df_supported = supported_candidates(df_parsed)

    assert len(df_supported) == 8
    assert set(df_supported["template"]) == set(TEMPLATE_TO_RULE.keys())
    assert df_supported["simulator_supported"].all()


def test_supported_candidates_raises_if_nothing_survives():
    df_raw = pd.DataFrame([_minerful_row("Init", ["X"]), _minerful_row("End", ["X"])])
    df_parsed = parse_candidates(df_raw)  # no se cae aca -- se conservan ambas filas
    assert len(df_parsed) == 2

    with pytest.raises(RuntimeError):
        supported_candidates(df_parsed)  # aca si, porque ninguna es simulable


@pytest.mark.parametrize("template", sorted(DSL_FORMAT.keys()))
def test_write_rule_ini_roundtrips_through_extract_rules(tmp_path, template):
    """
    Escribe la regla y la vuelve a leer con la funcion EXISTENTE (sin modificar)
    support_modules.traces_evaluation.extract_rules -- valida que lo que este modulo
    escribe es exactamente lo que el pipeline LSTM va a parsear despues.
    """
    output_path = tmp_path / f"{template}.ini"
    is_unary = template in ("absence", "required")
    arg0, arg1 = "ActivityX", (None if is_unary else "ActivityY")

    write_rule_ini(template, arg0, arg1, output_path)
    assert output_path.exists()

    settings = extract_rules(path=str(output_path))

    assert settings["rule"] == TEMPLATE_TO_RULE[template]
    if is_unary:
        assert settings["path"] == [arg0]
    else:
        assert settings["path"] == [arg0, arg1]


def test_write_rule_ini_writes_to_exact_path_not_a_directory(tmp_path):
    """
    Test de regresion del bug de rutas de convertir_declare_to_declarative
    (confirmado en disco: data/0.logs/BPI_Challenge_2012rules.ini,
    RunningExamplerules.ini, PurchasingExamplerules.ini) -- un argumento con forma
    de nombre de archivo no debe tratarse como directorio.
    """
    output_path = tmp_path / "nested" / "my_rule.ini"
    result = write_rule_ini("precedence", "A", "B", output_path)
    assert result == output_path
    assert output_path.is_file()
    assert output_path.name == "my_rule.ini"


def test_top_support_candidate_picks_highest_combined_score():
    df_candidates = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "template": "coexistence",
                "rule": "coexistence",
                "arg0": "X",
                "arg1": "Y",
                "Support": 0.9,
                "Coverage": 0.9,
                "Confidence": 1.0,
            },
            {
                "candidate_id": "b",
                "template": "precedence",
                "rule": "precedence",
                "arg0": "X",
                "arg1": "Z",
                "Support": 0.3,
                "Coverage": 0.3,
                "Confidence": 0.8,
            },
        ]
    )
    winner = top_support_candidate(df_candidates)
    assert winner["candidate_id"] == "a"


def test_top_support_candidate_ignores_unsupported_even_if_higher_score():
    """
    Si por error se le pasa el pool COMPLETO (con no soportadas incluidas), debe
    filtrar defensivamente antes de elegir -- una regla que el simulador no puede
    imponer no puede ser el baseline Top-Support, aunque tenga el score mas alto.
    """
    df_candidates = pd.DataFrame(
        [
            {
                "candidate_id": "unsupported_but_high_score",
                "template": "atleast3",
                "rule": None,
                "simulator_supported": False,
                "arg0": "X",
                "arg1": None,
                "Support": 0.99,
                "Coverage": 0.99,
                "Confidence": 1.0,
            },
            {
                "candidate_id": "supported_lower_score",
                "template": "coexistence",
                "rule": "coexistence",
                "simulator_supported": True,
                "arg0": "X",
                "arg1": "Y",
                "Support": 0.5,
                "Coverage": 0.5,
                "Confidence": 0.9,
            },
        ]
    )
    winner = top_support_candidate(df_candidates)
    assert winner["candidate_id"] == "supported_lower_score"
