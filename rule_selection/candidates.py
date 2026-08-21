"""
Generacion del pool de candidatas (CLAUDE.md Seccion 12.2): MINERful corre UNA sola
vez sobre el log completo (no por traza, no por grupo, no en un loop).

Decision explicita (2026-08-10, a pedido del usuario): **no se descarta ninguna
candidata** en esta etapa, incluidas las plantillas que el simulador todavia no
sabe evaluar/imponer (`AtLeast3`, `AtMost3`, `ChainResponse`, etc.). `parse_candidates`
conserva el pool completo y marca con `simulator_supported` cuales de las 8
plantillas que soportan `rule_selection.trace_index`/`support_modules.traces_evaluation`
son imponibles HOY -- si mas adelante se agrega soporte a una plantilla nueva, sus
candidatas ya quedan registradas sin tener que volver a correr MINERful sobre el
log. El resto del pipeline (uplift, screening, comparacion final) SI debe operar
solo sobre el subconjunto soportado (`supported_candidates`), porque
`evaluate_condition_list` no sabe chequear las demas evento por evento durante la
alucinacion LSTM, y `write_rule_ini` no tiene sintaxis DSL para ellas (CLAUDE.md
12.2, "no tocar el simulador").

No se toca `MINERful/modelo_declarativo_enriquecedio_v2.py`: se reutilizan
`carga_df_to_format_xes` y `ejecutar_minerful` tal cual, y solo se reimplementa el
*formateo* de string del diccionario `mapeo` de `convertir_declare_to_declarative`
(no su logica de escritura de archivo, que tiene un bug de rutas confirmado -- ver
`write_rule_ini`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from MINERful.modelo_declarativo_enriquecedio_v2 import carga_df_to_format_xes, ejecutar_minerful
from rule_selection._util import chdir as _chdir

# Plantilla MINERful (minuscula) -> valor `rule` que espera
# TraceIndex.evaluate / evaluate_condition_list. NotChainSuccession queda en
# PascalCase a proposito -- asi lo espera el codigo existente (inconsistente con
# los demas valores, pero es lo que hay).
TEMPLATE_TO_RULE: dict[str, str] = {
    "absence": "not_allowed",
    "required": "required",
    "response": "eventually",
    "alternateresponse": "directly",
    "precedence": "precedence",
    "succession": "succession",
    "coexistence": "coexistence",
    "notchainsuccession": "NotChainSuccession",
}

# Mismas 8 claves -> sintaxis del DSL `.ini` que consume `extract_rules`/el filtro de
# alucinacion LSTM. Copiado del `mapeo` de `convertir_declare_to_declarative`
# (MINERful/modelo_declarativo_enriquecedio_v2.py, lineas 664-673).
DSL_FORMAT: dict[str, str] = {
    "absence": "^{A}",
    "required": "{A}",
    "response": "{A} >> * >> {B}",
    "alternateresponse": "{A} >> {B}",
    "precedence": "{A} >> | >> {B}",
    "succession": "{A} >> # >> {B}",
    "coexistence": "{A} >> + >> {B}",
    "notchainsuccession": "{A} >> - >> {B}",
}

_CONSTRAINT_RE = re.compile(r"(\w+)\((.*)\)")


def export_log_to_xes(df_log: pd.DataFrame, xes_path: Path) -> Path:
    """Envoltorio sobre `carga_df_to_format_xes`, sin modificarla."""
    carga_df_to_format_xes(df_log.copy(), str(xes_path))
    return Path(xes_path)


def run_minerful_once(
    xes_path: Path,
    minerful_root: Path,
    csv_out: Path,
    s_min: float = 0.05,
    c_min: float = 0.8,
) -> pd.DataFrame:
    """
    Corre MINERful UNA sola vez sobre el log completo exportado en `xes_path`.
    Reutiliza `ejecutar_minerful` sin modificarla -- solo maneja el cwd relativo que
    esa funcion necesita para encontrar `MINERful.jar`/`lib/`.
    """
    xes_path = Path(xes_path).resolve()
    csv_out = Path(csv_out).resolve()
    minerful_root = Path(minerful_root).resolve()

    with _chdir(minerful_root):
        df = ejecutar_minerful(str(xes_path), str(csv_out), support=s_min, confidence=c_min)

    if df is None or df.empty:
        raise RuntimeError(
            f"MINERful no devolvio candidatas (s_min={s_min}, c_min={c_min}). "
            f"Revisar {csv_out} y stderr de java."
        )
    return df


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(text)).strip("_")


def parse_candidates(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parsea TODAS las restricciones descubiertas por MINERful -- no descarta ninguna
    por plantilla. Cada fila queda marcada con `simulator_supported` (True si su
    plantilla es una de las 8 que hoy soportan `TraceIndex.evaluate`/la alucinacion
    LSTM, False en cualquier otro caso, ej. `AtLeast3`, `AtMost3`, `Init`, `End`,
    `ChainResponse`). Solo se descartan filas que ni siquiera se pueden parsear
    (forma `Nombre(args)` invalida, o con 0/3+ argumentos -- eso es un problema de
    forma, no de soporte).

    Columnas del resultado: candidate_id, Constraint, template (plantilla MINERful
    en minuscula), rule (valor que espera TraceIndex.evaluate; None si no soportada),
    simulator_supported (bool), arg0, arg1 (None si la plantilla es unaria),
    Support, Coverage, Confidence.
    """
    rows = []
    for _, row in df_raw.iterrows():
        constraint = str(row["Constraint"]).strip()
        match = _CONSTRAINT_RE.match(constraint)
        if not match:
            continue

        template_name = match.group(1).lower()
        rule = TEMPLATE_TO_RULE.get(template_name)  # None si el simulador no la soporta hoy

        args = [a.strip().strip("[]") for a in match.group(2).split(",")]
        if len(args) == 1:
            arg0, arg1 = args[0], None
        elif len(args) == 2:
            arg0, arg1 = args[0], args[1]
        else:
            continue  # forma inesperada (0 o 3+ argumentos), se descarta igual

        candidate_id = f"{template_name}_{_slug(arg0)}"
        if arg1 is not None:
            candidate_id += f"_{_slug(arg1)}"

        rows.append(
            {
                "candidate_id": candidate_id,
                "Constraint": constraint,
                "template": template_name,
                "rule": rule,
                "simulator_supported": rule is not None,
                "arg0": arg0,
                "arg1": arg1,
                "Support": row.get("Support"),
                "Coverage": row.get("Coverage"),
                "Confidence": row.get("Confidence"),
            }
        )

    df_parsed = pd.DataFrame(rows)
    if df_parsed.empty:
        raise RuntimeError(
            "Ninguna restriccion devuelta por MINERful se pudo parsear -- revisar la salida cruda."
        )
    return df_parsed.drop_duplicates(subset="candidate_id").reset_index(drop=True)


def supported_candidates(df_candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Subconjunto de `df_candidates` que el simulador puede evaluar e imponer HOY
    (las 8 plantillas soportadas). El resto del pipeline (uplift, screening,
    comparacion final) debe operar solo sobre este subconjunto -- las demas no
    tienen `rule` para `TraceIndex.evaluate` ni sintaxis en `DSL_FORMAT` para
    `write_rule_ini`.
    """
    df_supported = df_candidates[df_candidates["simulator_supported"]].reset_index(drop=True)
    if df_supported.empty:
        raise RuntimeError(
            "Ninguna candidata cae dentro de las 8 plantillas que el simulador soporta hoy "
            f"({sorted(TEMPLATE_TO_RULE)}). Revisar la salida cruda de MINERful."
        )
    return df_supported


def generate_candidates(
    df_log: pd.DataFrame,
    minerful_root: Path,
    work_dir: Path,
    s_min: float = 0.05,
    c_min: float = 0.8,
) -> pd.DataFrame:
    """
    Paso 1 completo (CLAUDE.md 12.2): exporta el log a XES, corre MINERful una sola
    vez, y parsea TODAS las restricciones devueltas (ver `parse_candidates` -- no se
    descarta ninguna por plantilla, solo se marca cuales son simulables hoy).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    xes_path = work_dir / "candidatas_log.xes"
    csv_out = work_dir / "candidatas_minerful.csv"

    export_log_to_xes(df_log, xes_path)
    df_raw = run_minerful_once(xes_path, minerful_root, csv_out, s_min=s_min, c_min=c_min)
    return parse_candidates(df_raw)


def top_support_candidate(df_candidates: pd.DataFrame) -> pd.Series:
    """
    Regla con mayor Support+Coverage+Confidence entre las candidatas YA generadas
    (mismo criterio que `generar_modelo_global`, lineas 948-953) -- sin volver a
    correr MINERful. Es el baseline Top-Support (CLAUDE.md 12.6), reformulado como
    brazo de ablacion del criterio de seleccion, no como tecnica externa.

    Si `df_candidates` trae la columna `simulator_supported` (ej. si por error se
    le pasa el pool completo de `parse_candidates` en vez de
    `supported_candidates(...)`), se filtra defensivamente antes de elegir -- una
    regla no simulable no puede ser el baseline Top-Support, porque despues hay que
    escribirle un `.ini` y simularla como uno de los 4 brazos finales.
    """
    scored = df_candidates.copy()
    if "simulator_supported" in scored.columns:
        scored = scored[scored["simulator_supported"]]
        if scored.empty:
            raise RuntimeError(
                "No hay candidatas soportadas por el simulador para elegir Top-Support."
            )
    scored["score"] = scored["Support"] + scored["Coverage"] + scored["Confidence"]
    return scored.sort_values("score", ascending=False).iloc[0]


def write_rule_ini(template: str, arg0: str, arg1: str | None, output_path: Path) -> Path:
    """
    Escribe un `.ini` con la sintaxis que consume `extract_rules`/la alucinacion
    LSTM, EXACTAMENTE en `output_path` -- a diferencia de
    `convertir_declare_to_declarative`, que siempre trata su argumento como
    directorio y hardcodea el nombre a `rules.ini`. Ese bug ya se disparo antes: hay
    en el repo `data/0.logs/BPI_Challenge_2012rules.ini`,
    `data/0.logs/RunningExamplerules.ini`, `data/0.logs/PurchasingExamplerules.ini`
    -- nombres concatenados sin separador, evidencia directa.

    `template` es la plantilla MINERful en minuscula (clave de DSL_FORMAT, ej.
    "precedence", "response", "notchainsuccession") -- no el valor `rule` que usa
    TraceIndex (que para 3 de las 8 plantillas tiene un nombre distinto).
    """
    dsl = DSL_FORMAT.get(template)
    if dsl is None:
        raise ValueError(f"Plantilla sin formato DSL: {template!r} (validas: {sorted(DSL_FORMAT)})")

    path_str = dsl.format(A=arg0, B=arg1 or "")
    contenido = f"[RULES]\npath = {path_str}\nvariation = =1"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(contenido, encoding="utf-8")
    return output_path
