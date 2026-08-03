"""
Metricas de evaluacion para logs de simulacion de procesos de negocio (BPS).
Basadas en: Graziosi et al. (2024) "Generating the traces you need: A conditional
generative model for process mining data". arXiv:2411.02131

Metricas implementadas
----------------------
RED  - Relative Event Distribution   : EMD entre distribuciones de tiempos inter-evento
CTD  - Cycle Time Distribution       : EMD entre distribuciones de tiempos de ciclo
2GD  - 2-Gram Distance               : EMD (TVD) entre distribuciones de bigramas
CONF - Conformance Score             : conformidad promedio contra reglas DECLARE (soporte >= 90%)

Convencion de distancias
------------------------
RED  : valor en [0, 100] (EMD Wasserstein-1 entre distribuciones de posicion ordinal relativa, en %), menor es mejor
CTD  : valor en segundos (EMD Wasserstein-1 entre distribuciones de tiempo de ciclo), menor es mejor
2GD  : valor en [0, 1] (TVD = EMD con coste unitario categorico), menor es mejor
CONF : valor en [0, 1], MAYOR es mejor
"""

import configparser
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wasserstein_distance as _scipy_emd
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn(
        "scipy no esta disponible. Se usara una aproximacion del EMD.",
        ImportWarning,
        stacklevel=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalizacion de columnas
# ─────────────────────────────────────────────────────────────────────────────

_ALIASES: Dict[str, List[str]] = {
    "caseid": ["caseid", "case_id", "case:concept:name", "Case ID", "CaseID", "case"],
    "task":   ["task", "activity", "concept:name", "Activity", "Task"],
    "start_timestamp": ["start_timestamp", "start_time", "StartTimestamp"],
    "end_timestamp":   ["end_timestamp", "end_time", "EndTimestamp", "CompleteTimestamp"],
}


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in _ALIASES[target]:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def load_log(path: str) -> pd.DataFrame:
    """
    Carga un CSV de log de eventos y normaliza los nombres de columnas a:
    caseid, task, start_timestamp, end_timestamp.

    Acepta .csv y .csv.gz. Timestamps con zona horaria se convierten a UTC naive.
    """
    df = pd.read_csv(path, compression="infer")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    rename = {}
    for target in ("caseid", "task", "start_timestamp", "end_timestamp"):
        src = _find_col(df, target)
        if src and src != target:
            rename[src] = target
    df = df.rename(columns=rename)

    for col in ("start_timestamp", "end_timestamp"):
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.tz_convert(None)

    missing = [c for c in ("caseid", "task") if c not in df.columns]
    if missing:
        raise ValueError(
            f"No se encontraron columnas requeridas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: Earth Mover's Distance 1D
# ─────────────────────────────────────────────────────────────────────────────

def _emd_continuous(u: np.ndarray, v: np.ndarray) -> float:
    """
    EMD (Wasserstein-1) entre dos muestras de distribuciones continuas.
    Resultado en las mismas unidades que los valores de entrada.
    """
    if len(u) == 0 or len(v) == 0:
        return float("nan")
    if _HAS_SCIPY:
        return float(_scipy_emd(u, v))
    # Fallback: area entre CDFs en grilla comun
    n = max(len(u), len(v))
    grid = np.linspace(0, 1, n)
    u_i = np.interp(grid, np.linspace(0, 1, len(u)), np.sort(u))
    v_i = np.interp(grid, np.linspace(0, 1, len(v)), np.sort(v))
    return float(np.mean(np.abs(u_i - v_i)))


def _emd_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """
    EMD con coste unitario entre categorias = Total Variation Distance.
    Resultado en [0, 1]: 0 = distribuciones identicas, 1 = maxima diferencia.
    Equivalente al EMD de Wasserstein-1 cuando la distancia entre cualquier par
    de categorias distintas es 1.
    """
    return float(0.5 * np.sum(np.abs(p - q)))


# ─────────────────────────────────────────────────────────────────────────────
# Extraccion de trazas como listas de tareas
# ─────────────────────────────────────────────────────────────────────────────

def _get_sorted_traces(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Devuelve un dict {case_id: [task1, task2, ...]} ordenado por start_timestamp.
    """
    sort_col = "start_timestamp" if "start_timestamp" in df.columns else "end_timestamp"
    traces: Dict[str, List[str]] = {}
    for case_id, grp in df.sort_values(["caseid", sort_col]).groupby("caseid", sort=False):
        traces[str(case_id)] = grp["task"].tolist()
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# RED - Relative Event Distribution
# ─────────────────────────────────────────────────────────────────────────────

def compute_red(df_ref: pd.DataFrame, df_sim: pd.DataFrame) -> float:
    """
    Relative Event Distribution (RED).

    Para cada evento en cada traza, calcula su posicion ordinal relativa
    expresada como porcentaje de la longitud de la traza:

        pos_relativa = (indice_ordinal / longitud_traza) * 100

    donde indice_ordinal va de 0 (primer evento) a L-1 (ultimo evento)
    y longitud_traza = L = numero de eventos en la traza.

    Los valores resultantes estan en [0, 100] (% de posicion en la traza).
    Se recogen todos los valores de todos los eventos de todos los casos,
    se construye una distribucion y se aplica EMD (Wasserstein-1) entre
    la distribucion del log de referencia y la del log simulado.

    Esta definicion es compatible con Graziosi et al. (2024) / Camargo et al.,
    y permite comparacion directa de valores numericos entre trabajos.

    Resultado en [0, 100] (menor = distribuciones de posicion mas similares).
    """
    def _relative_positions(df: pd.DataFrame) -> np.ndarray:
        sort_col = "start_timestamp" if "start_timestamp" in df.columns else "end_timestamp"

        positions: List[float] = []
        for _, grp in df.sort_values(sort_col).groupby("caseid", sort=False):
            L = len(grp)
            if L == 0:
                continue
            for k in range(L):
                positions.append((k / L) * 100.0)

        return np.array(positions, dtype=float)

    pos_ref = _relative_positions(df_ref)
    pos_sim = _relative_positions(df_sim)
    return _emd_continuous(pos_ref, pos_sim)


# ─────────────────────────────────────────────────────────────────────────────
# CTD - Cycle Time Distribution
# ─────────────────────────────────────────────────────────────────────────────

def compute_ctd(df_ref: pd.DataFrame, df_sim: pd.DataFrame) -> float:
    """
    Cycle Time Distribution (CTD).

    Calcula el tiempo de ciclo de cada traza:
        cycle_time = max(end_timestamp) - min(start_timestamp)

    Compara las distribuciones de tiempos de ciclo con EMD (Wasserstein-1).
    Resultado en segundos (menor = mas similar).

    Referencia: Graziosi et al. (2024), Sec. de metricas temporales.
    """
    def _cycle_times(df: pd.DataFrame) -> np.ndarray:
        g = df.groupby("caseid").agg(
            s=("start_timestamp", "min"),
            e=("end_timestamp", "max"),
        )
        ct = (g["e"] - g["s"]).dt.total_seconds().dropna()
        return ct.values

    ct_ref = _cycle_times(df_ref)
    ct_sim = _cycle_times(df_sim)
    return _emd_continuous(ct_ref, ct_sim)


# ─────────────────────────────────────────────────────────────────────────────
# 2GD - 2-Gram Distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_2gd(df_ref: pd.DataFrame, df_sim: pd.DataFrame) -> float:
    """
    2-Gram Distance (2GD).

    Extrae bigramas (pares de actividades consecutivas) de cada traza,
    construye distribuciones de probabilidad y aplica EMD con coste unitario
    entre categorias (= Total Variation Distance).

    Resultado en [0, 1]: 0 = distribuciones identicas, 1 = maxima diferencia.

    Referencia: Graziosi et al. (2024), Sec. de metricas flujo de control.
    """
    def _bigram_dist(df: pd.DataFrame) -> Tuple[List, np.ndarray]:
        sort_col = "start_timestamp" if "start_timestamp" in df.columns else "end_timestamp"
        bg: Counter = Counter()
        for _, grp in df.sort_values(["caseid", sort_col]).groupby("caseid", sort=False):
            tasks = grp.sort_values(sort_col)["task"].tolist()
            bg.update(zip(tasks[:-1], tasks[1:]))
        return bg

    bg_ref = _bigram_dist(df_ref)
    bg_sim = _bigram_dist(df_sim)

    all_keys = sorted(set(bg_ref) | set(bg_sim), key=str)
    if not all_keys:
        return 0.0

    v_ref = np.array([bg_ref.get(k, 0) for k in all_keys], dtype=float)
    v_sim = np.array([bg_sim.get(k, 0) for k in all_keys], dtype=float)

    p = v_ref / v_ref.sum() if v_ref.sum() > 0 else v_ref
    q = v_sim / v_sim.sum() if v_sim.sum() > 0 else v_sim

    return _emd_discrete(p, q)


# ─────────────────────────────────────────────────────────────────────────────
# CONF - Conformance Score
# Descubrimiento: MINERful (via modelo_declarativo_enriquecedio.py)
# Verificacion:   pm4py.conformance_declare
# ─────────────────────────────────────────────────────────────────────────────

import os as _os
import re as _re
import sys as _sys
from pathlib import Path as _Path

# Todos los tipos DECLARE disponibles (para CLI y validacion externa)
AVAILABLE_RULES: List[str] = [
    "existence", "absence", "exactly_one", "init", "end",
    "responded_existence", "coexistence", "response", "precedence",
    "succession", "altresponse", "altprecedence", "altsuccession",
    "chainresponse", "chainprecedence", "chainsuccession",
    "not_co_existence", "not_succession", "not_chainsuccession",
]

# Mapeo de nombres de templates MINERful → pm4py.
# Solo se incluyen los templates que pm4py.conformance_declare soporta
# (los templates "negativos" de MINERful como NotSuccession no estan soportados
# por pm4py 2.7.x y causan KeyError interno).
_MINERFUL_TO_PM4PY: Dict[str, str] = {
    "existence":           "existence",
    "absence":             "absence",
    "exactlyone":          "exactly_one",
    "exactly1":            "exactly_one",
    # atmost1 omitido: "at most 1" != "exactly 1" en pm4py; no hay equivalente directo
    "required":            "existence",
    "init":                "init",
    "respondedexistence":  "responded_existence",
    "coexistence":         "coexistence",
    "response":            "response",
    "precedence":          "precedence",
    "succession":          "succession",
    "alternateresponse":   "altresponse",
    "alternateprecedence": "altprecedence",
    "alternatesuccession": "altsuccession",
    "chainresponse":       "chainresponse",
    "chainprecedence":     "chainprecedence",
    "chainsuccession":     "chainsuccession",
    # Templates negativos de MINERful: NO mapeados (pm4py 2.7.x no los soporta)
    # "notsuccession", "notcoexistence", "notchainsuccession", etc. → omitidos
}

# NOTA (verificado 2026-08-xx contra datos reales de MINERful en este repo y
# contra el codigo fuente de pm4py `classic.py::__check_alt_succession` /
# `__check_chain_succession`): el string 'Constraint' de MINERful NO tiene un
# orden de argumentos consistente entre familias de plantillas — la familia
# Precedence (Precedence, NotPrecedence, ChainPrecedence, NotChainPrecedence...)
# escribe el string como Template(Target, Activation) mientras que la familia
# Response/Succession lo escribe como Template(Activation, Target). Ejemplo real
# de este repo: 'Precedence(A_SUBMITTED, A_DECLINED)' con columnas
# Activation='[A_DECLINED]', Target='[A_SUBMITTED]' — el primer argumento del
# string es el TARGET, no la activacion. Parsear el string por posicion (como
# hacia el codigo anterior con _REVERSE_ARGS) adivina mal para toda la familia
# Precedence. La correccion es usar directamente las columnas 'Activation' y
# 'Target' que MINERful ya entrega sin ambiguedad — ver _minerful_to_pm4py_model.
# pm4py en si mismo SI usa un orden uniforme (act_couple[0]=activacion,
# act_couple[1]=target) para todas las plantillas binarias, incluidas
# altsuccession/chainsuccession — no requiere ninguna inversion especial.


def _df_to_xes_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte nuestro formato (caseid, task, start_timestamp, end_timestamp)
    al formato XES estandar que espera carga_df_to_format_xes.
    """
    ts_col = "start_timestamp" if "start_timestamp" in df.columns else "end_timestamp"
    df_xes = pd.DataFrame({
        "case:concept:name":    df["caseid"].astype(str),
        "concept:name":         df["task"].astype(str),
        "time:timestamp":       df[ts_col],
        "lifecycle:transition": "complete",
        "org:resource": (
            df["resource"].fillna("UNKNOWN").astype(str)
            if "resource" in df.columns else "UNKNOWN"
        ),
    })
    return df_xes.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)


def _run_minerful(df_ref: pd.DataFrame, support_threshold: float) -> Optional[pd.DataFrame]:
    """
    Exporta df_ref a XES y ejecuta MINERful para descubrir reglas DECLARE.
    Retorna el DataFrame de reglas, o None si MINERful no esta disponible.
    """
    # Ubicar el directorio MINERful relativo a este archivo (evaluacion/metrics.py)
    minerful_dir = _Path(__file__).resolve().parent.parent / "MINERful"
    if not minerful_dir.exists():
        warnings.warn(f"Directorio MINERful no encontrado: {minerful_dir}")
        return None

    # Importar funciones de modelo_declarativo_enriquecedio.py
    if str(minerful_dir) not in _sys.path:
        _sys.path.insert(0, str(minerful_dir))
    try:
        from modelo_declarativo_enriquecedio import (
            carga_df_to_format_xes, ejecutar_minerful,
        )
    except ImportError as exc:
        warnings.warn(f"No se pudo importar modelo_declarativo_enriquecedio: {exc}")
        return None

    # Rutas absolutas para los archivos temporales (dentro del dir MINERful)
    xes_path = str(minerful_dir / "_conf_temp.xes")
    csv_path = str(minerful_dir / "_conf_temp.csv")
    original_cwd = _os.getcwd()

    try:
        # Convertir a formato XES y exportar
        carga_df_to_format_xes(_df_to_xes_cols(df_ref).copy(), xes_path)

        # MINERful.jar usa rutas relativas → cambiar al directorio donde esta el JAR
        _os.chdir(str(minerful_dir))

        # Pasamos umbrales muy bajos para obtener TODAS las reglas descubiertas
        # y filtrar nosotros por 'Trace support' (fraccion de trazas), que es la
        # definicion correcta para CONF segun el paper.
        # El parametro -s de MINERful filtra por 'Support' (por evento), NO por traza.
        rules_df = ejecutar_minerful(
            input_log=xes_path,
            output_csv=csv_path,
            support=0.0,
            confidence=0.0,
        )
        return rules_df

    except Exception as exc:
        warnings.warn(f"MINERful fallo: {exc}")
        return None
    finally:
        _os.chdir(original_cwd)
        for p in [xes_path, csv_path]:
            try:
                if _os.path.exists(p):
                    _os.remove(p)
            except Exception:
                pass


def _minerful_to_pm4py_model(rules_df: pd.DataFrame, support_threshold: float = 0.9) -> Dict:
    """
    Convierte el DataFrame de salida de MINERful al formato de modelo DECLARE
    que espera pm4py.conformance_declare:
        { template_name: { key: {'support': N, 'confidence': N} } }
    donde key es str (unaria) o tuple (binaria).

    Filtra por 'Trace support' (fraccion de TRAZAS satisfaciendo la regla),
    que es la definicion correcta segun el paper.
    MINERful distingue 'Support' (por evento) de 'Trace support' (por traza).
    """
    cols_lower = {c.lower().strip(): c for c in rules_df.columns}
    constraint_col    = cols_lower.get("constraint")
    activation_col    = cols_lower.get("activation")
    target_col        = cols_lower.get("target")
    trace_support_col = cols_lower.get("trace support")   # columna clave
    support_col       = cols_lower.get("support")         # fallback

    if constraint_col is None:
        warnings.warn("El CSV de MINERful no tiene columna 'Constraint'.")
        return {}
    if activation_col is None or target_col is None:
        warnings.warn(
            "El CSV de MINERful no tiene columnas 'Activation'/'Target' — "
            "se cae a parsear el string 'Constraint' por posicion (menos "
            "confiable para la familia Precedence, ver comentario arriba).")

    def _clean_activity(raw) -> str:
        s = str(raw).strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        return s.strip().strip("'\"").strip()

    model: Dict = {}

    for _, row in rules_df.iterrows():
        raw = str(row[constraint_col]).strip("'").strip()

        # Usar 'Trace support' si existe; si no, caer en 'Support'
        if trace_support_col:
            trace_sup = float(row[trace_support_col])
        elif support_col:
            trace_sup = float(row[support_col])
        else:
            trace_sup = 1.0

        # Filtrar por fraccion de trazas
        if trace_sup < support_threshold:
            continue

        m = _re.match(r"(\w+)\((.+)\)", raw)
        if not m:
            continue

        pm4py_tmpl = _MINERFUL_TO_PM4PY.get(m.group(1).lower())
        if pm4py_tmpl is None:
            continue

        counts = {
            "support":    int(round(trace_sup * 100)),
            "confidence": int(round(trace_sup * 100)),
        }

        if pm4py_tmpl not in model:
            model[pm4py_tmpl] = {}

        if activation_col is not None and target_col is not None:
            # Fuente de verdad: columnas Activation/Target de MINERful, sin
            # ambiguedad de orden entre familias de plantillas.
            activation = _clean_activity(row[activation_col])
            target     = _clean_activity(row[target_col])
            if not activation:
                # Plantillas unarias (existence, absence, init, end...): el
                # unico argumento relevante queda en Target.
                model[pm4py_tmpl][target] = counts
            else:
                model[pm4py_tmpl][(activation, target)] = counts
        else:
            # Fallback: orden posicional del string 'Constraint' (pm4py usa
            # orden uniforme act_couple[0]=activacion, act_couple[1]=target
            # para todas las plantillas binarias — sin inversion especial).
            args = [a.strip() for a in m.group(2).split(",")]
            if len(args) == 1:
                model[pm4py_tmpl][args[0]] = counts
            elif len(args) >= 2:
                model[pm4py_tmpl][(args[0], args[1])] = counts

    # pm4py 2.7.x requiere que las plantillas base existan en el modelo
    # cuando sus variantes alternas estan presentes (acceso interno a model['response'], etc.)
    _alt_deps = {
        "altresponse":   "response",
        "altprecedence": "precedence",
        "altsuccession": "succession",
    }
    for alt, base in _alt_deps.items():
        if alt in model and base not in model:
            model[base] = {}

    return model


def discover_conf_model(
    df_ref: pd.DataFrame, support_threshold: float = 0.9
) -> Tuple[Dict, List[str]]:
    """
    Fase 1 del pipeline CONF: descubre el modelo DECLARE con MINERful.

    Retorna (pm4py_model, constraint_strings) donde:
      pm4py_model        : dict en formato pm4py.conformance_declare
      constraint_strings : lista de strings de restricciones (para el reporte)

    Usa MINERful si esta disponible; de lo contrario retorna ({}, []).
    """
    rules_df = _run_minerful(df_ref, support_threshold)

    if rules_df is None or rules_df.empty:
        warnings.warn(
            "MINERful no genero reglas. CONF no estara disponible."
        )
        return {}, []

    pm4py_model = _minerful_to_pm4py_model(rules_df, support_threshold=support_threshold)

    # Reconstruir strings de las restricciones efectivamente mapeadas al modelo pm4py
    constraint_strings: List[str] = []
    for tmpl, cdict in pm4py_model.items():
        for key in cdict:
            if isinstance(key, tuple):
                constraint_strings.append(f"{tmpl}({key[0]}, {key[1]})")
            else:
                constraint_strings.append(f"{tmpl}({key})")

    return pm4py_model, constraint_strings


def compute_conf(
    df_ref: pd.DataFrame,
    df_sim: pd.DataFrame,
    support_threshold: float = 0.9,
    cached_model: Optional[Dict] = None,
) -> Tuple[float, List[str]]:
    """
    Conformance Score (CONF).

    Pipeline:
    1. Descubrimiento: MINERful descubre restricciones DECLARE del log de
       referencia con soporte >= support_threshold.
    2. Conformance:    pm4py.conformance_declare verifica cada traza del log
       simulado contra esas restricciones.
    3. Score final:    promedio de dev_fitness por traza (0-1, MAYOR es mejor).

    Parametros
    ----------
    df_ref            : log de referencia (origen de las reglas DECLARE)
    df_sim            : log a evaluar
    support_threshold : umbral de soporte para MINERful (default 0.90)
    cached_model      : modelo pm4py pre-calculado (evita re-ejecutar MINERful
                        en evaluaciones por lote; obtenerlo con discover_conf_model)

    Retorna
    -------
    (conf_score, constraint_strings)
      conf_score         : float en [0,1]
      constraint_strings : lista de restricciones DECLARE usadas (para n_rules)
    """
    import pm4py

    # Fase 1: obtener el modelo
    if cached_model is None:
        cached_model, constraint_strings = discover_conf_model(df_ref, support_threshold)
    else:
        # Si el modelo ya esta cacheado, reconstruir los strings solo para el reporte
        constraint_strings = []
        for tmpl, cdict in cached_model.items():
            for key in cdict:
                if isinstance(key, tuple):
                    constraint_strings.append(f"{tmpl}({key[0]}, {key[1]})")
                else:
                    constraint_strings.append(f"{tmpl}({key})")

    if not cached_model:
        return float("nan"), []

    # Fase 2: proyectar df_sim en trazas de actividades, agrupando por caseid
    # en orden de timestamp para cada caso.
    # NO se usa pm4py.conformance_declare directamente con DataFrame porque
    # project_on_event_attribute ordena los case IDs lexicograficamente (bug
    # con IDs numericos: '1','10','11',...,'2') mezclando trazas de distintos casos.
    ts_col = "start_timestamp" if "start_timestamp" in df_sim.columns else "end_timestamp"
    projected_log: List[List[str]] = []
    for _, grp in df_sim.sort_values(["caseid", ts_col]).groupby("caseid", sort=False):
        projected_log.append(grp.sort_values(ts_col)["task"].tolist())

    # Fase 3: conformance checking con pm4py apply_list (nivel bajo, evita el bug)
    try:
        from pm4py.algo.conformance.declare.variants.classic import apply_list
        results = apply_list(projected_log, cached_model)
        conf_score = float(np.mean([r.get("dev_fitness", 0.0) for r in results]))
        return conf_score, constraint_strings

    except Exception as exc:
        warnings.warn(f"pm4py conformance_declare fallo: {exc}")
        return float("nan"), constraint_strings


# Alias para compatibilidad con codigo existente
def mine_declare_rules(df: pd.DataFrame, support_threshold: float = 0.9) -> List[str]:
    """Alias: ejecuta discover_conf_model y retorna la lista de strings de restricciones."""
    _, constraint_strings = discover_conf_model(df, support_threshold)
    return constraint_strings


# ─────────────────────────────────────────────────────────────────────────────
# Parseo de rules.ini (para uso legacy / CONF manual)
# ─────────────────────────────────────────────────────────────────────────────

_OP_TO_RULE = {
    "+": "co_existence",
    "*": "response",
    "|": "precedence",
    "#": "succession",
    "-": "not_succession",
}


def parse_rules_ini(path: str) -> Tuple[str, str, Optional[str]]:
    """
    Lee un rules.ini y devuelve (tipo_regla, act_a, act_b).
    Compatible con el formato del proyecto (usado en traces_evaluation.py).
    """
    config = configparser.ConfigParser()
    config.read(path)
    raw   = config["RULES"]["path"]
    parts = [x.strip() for x in raw.split(">>")]

    if "^" in raw:
        rule = "absence"
    elif ">>" not in raw:
        rule = "existence"
    else:
        rule = "chain_response"
        for op, rtype in _OP_TO_RULE.items():
            if op in parts:
                rule = rtype
                break

    acts  = [p for p in parts if p not in set(_OP_TO_RULE) | {">>"}]
    act_a = acts[0] if acts else None
    act_b = acts[1] if len(acts) > 1 else None
    return rule, act_a, act_b
