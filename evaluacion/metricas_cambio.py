"""
Metricas de calidad del comportamiento generado bajo un cambio declarativo phi.

Complementan a metrics.py (RED, CTD, 2GD, CONF) con las dimensiones de la
metodologia de evaluacion de RULE-GAN:

  Fidelidad      : distancia entre frecuencias de actividades; fidelidad en lo
                   que phi no toca (2GD y frecuencias sobre la traza sin las
                   actividades de phi)
  Novedad        : novedad por traza y por variante; novedad no pedida (la parte
                   de la traza que phi no toca es nueva); novedad validada (lo
                   nuevo aparece en un log de referencia, p.ej. el hold-out)
  Control        : proporcion de trazas que cumplen phi frente a la pedida;
                   tasa de cumplimiento por clase si el log trae la clase pedida
  Plausibilidad  : conformidad con las restricciones DECLARE minadas que phi no
                   contradice (media y estricta), por alcance local, vecindario y
                   global; compatibilidad rol-actividad
  Diversidad     : distancia de Damerau-Levenshtein promedio entre pares de trazas
  Novedad valida : % de trazas nuevas que cumplen phi y no violan ninguna
                   restriccion plausible

La regla phi es el dict que devuelve support_modules.traces_evaluation.extract_rules:
    {'rule': <tipo>, 'path': [actividades]}
y se evalua con evaluate_condition_list, la misma semantica que usan los
generadores (LSTM de Dynamics y RULE-GAN).

El modelo DECLARE de referencia tiene el formato de pm4py.conformance_declare
({plantilla: {clave: conteos}}) y se obtiene con metrics.discover_conf_model
sobre el log de ENTRENAMIENTO.
"""

import itertools
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from evaluacion.metrics import _get_sorted_traces, compute_2gd
except ImportError:
    from metrics import _get_sorted_traces, compute_2gd

from support_modules.traces_evaluation import evaluate_condition_list


Traza = Tuple[str, ...]

# Simbolo que representa "cualquier actividad ajena" en el chequeo de
# satisfacibilidad acotada. No puede coincidir con una actividad real.
_OTRA = "\x00OTRA\x00"


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def trazas(df: pd.DataFrame) -> Dict[str, Traza]:
    """{caso: tupla de actividades ordenadas por tiempo} (usa load_log de metrics)."""
    return {c: tuple(t) for c, t in _get_sorted_traces(df).items()}


def cumple_regla(traza: Sequence[str], regla: Dict) -> bool:
    """True si la traza cumple phi, con la semantica de evaluate_condition_list."""
    return bool(evaluate_condition_list(list(traza), {}, regla["path"], regla["rule"]))


def actividades_regla(regla: Dict) -> Set[str]:
    return set(regla["path"])


def proyectar(traza: Traza, quitar: Set[str]) -> Traza:
    """La traza sin las actividades de `quitar` (lo que phi no toca)."""
    return tuple(a for a in traza if a not in quitar)


def separar_por_regla(df: pd.DataFrame, regla: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(eventos de los casos que cumplen phi, eventos de los que no la cumplen)."""
    cumplen = {c for c, t in trazas(df).items() if cumple_regla(t, regla)}
    mask = df["caseid"].astype(str).isin(cumplen)
    return df[mask].copy(), df[~mask].copy()


def _tvd(c_ref: Counter, c_sim: Counter) -> float:
    claves = set(c_ref) | set(c_sim)
    if not claves:
        return 0.0
    n_ref, n_sim = sum(c_ref.values()), sum(c_sim.values())
    if n_ref == 0 or n_sim == 0:
        return float("nan")
    return float(0.5 * sum(abs(c_ref[k] / n_ref - c_sim[k] / n_sim) for k in claves))


# ─────────────────────────────────────────────────────────────────────────────
# Fidelidad
# ─────────────────────────────────────────────────────────────────────────────

def compute_freq_actividades(df_ref: pd.DataFrame, df_sim: pd.DataFrame) -> float:
    """
    Distancia de variacion total entre las frecuencias relativas de actividades
    (equivale a la mitad del ActError L1 de ProcessGAN). [0,1], menor es mejor.
    """
    return _tvd(Counter(df_ref["task"]), Counter(df_sim["task"]))


def compute_fidelidad_no_tocada(df_ref: pd.DataFrame, df_sim: pd.DataFrame,
                                regla: Dict) -> Dict[str, float]:
    """
    Fidelidad en lo que phi no toca: se quitan de ambos logs los eventos de las
    actividades de phi y se comparan 2GD y frecuencias de actividades.
    No se exige igualdad: un cambio puede alterar indirectamente el resto.
    """
    quitar = actividades_regla(regla)
    ref = df_ref[~df_ref["task"].isin(quitar)]
    sim = df_sim[~df_sim["task"].isin(quitar)]
    return {
        "2GD_no_tocada": compute_2gd(ref, sim),
        "freq_act_no_tocada": compute_freq_actividades(ref, sim),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Novedad
# ─────────────────────────────────────────────────────────────────────────────

def compute_novedad(df_train: pd.DataFrame, df_sim: pd.DataFrame) -> Dict[str, float]:
    """
    novedad_trazas    : % de trazas generadas cuya secuencia no esta en entrenamiento
    novedad_variantes : % de variantes distintas generadas que no estan en
                        entrenamiento (criterio de CVAE, Tabla III)
    """
    vistas = set(trazas(df_train).values())
    sim = list(trazas(df_sim).values())
    if not sim:
        return {"novedad_trazas": float("nan"), "novedad_variantes": float("nan")}
    variantes = set(sim)
    return {
        "novedad_trazas": float(np.mean([t not in vistas for t in sim])),
        "novedad_variantes": len(variantes - vistas) / len(variantes),
    }


def compute_novedad_no_pedida(df_train: pd.DataFrame, df_sim: pd.DataFrame,
                              regla: Dict) -> float:
    """
    % de trazas generadas cuya parte sin las actividades de phi no aparece en
    entrenamiento: comportamiento nuevo que phi no pidio. Menor es mejor.
    """
    quitar = actividades_regla(regla)
    vistas = {proyectar(t, quitar) for t in trazas(df_train).values()}
    sim = [proyectar(t, quitar) for t in trazas(df_sim).values()]
    return float(np.mean([t not in vistas for t in sim])) if sim else float("nan")


def compute_novedad_validada(df_train: pd.DataFrame, df_sim: pd.DataFrame,
                             df_ref: pd.DataFrame) -> Dict[str, float]:
    """
    De las trazas generadas nuevas respecto a entrenamiento, % cuya secuencia
    aparece en df_ref (p.ej. las trazas ocultas del hold-out): novedad que
    corresponde a comportamiento real.
    """
    vistas = set(trazas(df_train).values())
    ref = set(trazas(df_ref).values())
    nuevas = [t for t in trazas(df_sim).values() if t not in vistas]
    return {
        "n_nuevas": len(nuevas),
        "novedad_validada": float(np.mean([t in ref for t in nuevas])) if nuevas else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Control condicional
# ─────────────────────────────────────────────────────────────────────────────

def compute_control(df_sim: pd.DataFrame, regla: Dict,
                    p_pedida: Optional[float] = None,
                    col_pedido: Optional[str] = None,
                    valor_cumplir=None) -> Dict[str, float]:
    """
    prop_cumple       : % de trazas generadas que cumplen phi
    error_proporcion  : |prop_cumple - p_pedida| (si se da p_pedida)

    Si el log trae la clase pedida por caso (col_pedido, p.ej. 'label' en CVAE;
    valor_cumplir = valor que significa "debe cumplir phi"):
    tasa_cumple_pedidas   : de las trazas a las que se pidio cumplir, % que cumple
    tasa_evita_no_pedidas : de las trazas a las que no se pidio, % que no cumple

    Nota: si el generador descarta lo que no cumple, el log final cumple por
    construccion; para medir lo aprendido hay que pasar tambien el log ANTES
    de esa verificacion.
    """
    tr = trazas(df_sim)
    cumple = {c: cumple_regla(t, regla) for c, t in tr.items()}
    out = {"prop_cumple": float(np.mean(list(cumple.values()))) if cumple else float("nan")}
    if p_pedida is not None:
        out["p_pedida"] = p_pedida
        out["error_proporcion"] = abs(out["prop_cumple"] - p_pedida)
    if col_pedido is not None and col_pedido in df_sim.columns:
        pedido = (df_sim.groupby(df_sim["caseid"].astype(str))[col_pedido].first()
                  == valor_cumplir).to_dict()
        si = [cumple[c] for c in cumple if pedido.get(c)]
        no = [not cumple[c] for c in cumple if c in pedido and not pedido[c]]
        out["tasa_cumple_pedidas"] = float(np.mean(si)) if si else float("nan")
        out["tasa_evita_no_pedidas"] = float(np.mean(no)) if no else float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plausibilidad condicionada a phi
# ─────────────────────────────────────────────────────────────────────────────

def _actividades_restriccion(clave) -> Set[str]:
    return set(clave) if isinstance(clave, tuple) else {clave}


def _iter_restricciones(modelo: Dict):
    for plantilla, claves in modelo.items():
        for clave, conteos in claves.items():
            yield plantilla, clave, conteos


def _modelo_de(restricciones: Iterable[Tuple[str, object, dict]]) -> Dict:
    """Arma un modelo pm4py; agrega las plantillas base que pm4py 2.7 exige
    cuando hay variantes alternas (ver metrics._minerful_to_pm4py_model)."""
    modelo: Dict = {}
    for plantilla, clave, conteos in restricciones:
        modelo.setdefault(plantilla, {})[clave] = conteos
    for alt, base in (("altresponse", "response"), ("altprecedence", "precedence"),
                      ("altsuccession", "succession")):
        if alt in modelo and base not in modelo:
            modelo[base] = {}
    return modelo


def _apply_list(secuencias: List[List[str]], modelo: Dict) -> List[dict]:
    from pm4py.algo.conformance.declare.variants.classic import apply_list
    return apply_list(secuencias, modelo)


def restricciones_contradichas(modelo: Dict, regla: Dict, max_len: int = 5) -> Set[Tuple[str, object]]:
    """
    Restricciones del modelo que no pueden cumplirse junto con phi.

    Chequeo de satisfacibilidad acotado y por pares: una restriccion c se
    considera contradicha si ninguna traza de largo <= max_len, sobre el
    alfabeto act(c) U act(phi) U {otra}, cumple a la vez phi y c.

    Si c no comparte actividades con phi se asume compatible: las plantillas
    DECLARE sobre alfabetos disjuntos siempre admiten una traza que intercale
    ambas. Es una aproximacion (pares, largo acotado) y debe declararse asi.
    """
    act_phi = actividades_regla(regla)

    # phi solo mira sus propias actividades: cualquier otra es intercambiable.
    # Se evalua una vez sobre secuencias canonicas (actividades de phi + OTRA).
    canon = sorted(act_phi) + [_OTRA]
    phi_ok = {}
    for n in range(1, max_len + 1):
        for s in itertools.product(canon, repeat=n):
            phi_ok[s] = cumple_regla(s, regla)

    cache_alfabeto: Dict[Tuple[str, ...], List[List[str]]] = {}
    contradichas: Set[Tuple[str, object]] = set()
    for plantilla, clave, conteos in _iter_restricciones(modelo):
        act_c = _actividades_restriccion(clave)
        if not act_c & act_phi:
            continue
        alfabeto = tuple(sorted(act_c | act_phi)) + (_OTRA,)
        if alfabeto not in cache_alfabeto:
            secs = []
            for n in range(1, max_len + 1):
                for s in itertools.product(alfabeto, repeat=n):
                    if phi_ok[tuple(a if a in act_phi else _OTRA for a in s)]:
                        secs.append(list(s))
            cache_alfabeto[alfabeto] = secs
        secs = cache_alfabeto[alfabeto]
        modelo_c = _modelo_de([(plantilla, clave, conteos)])
        compatible = False
        for i in range(0, len(secs), 500):
            if any(not r["deviations"] for r in _apply_list(secs[i:i + 500], modelo_c)):
                compatible = True
                break
        if not compatible:
            contradichas.add((plantilla, clave))
    return contradichas


def filtrar_modelo(modelo: Dict, quitar: Set[Tuple[str, object]]) -> Dict:
    """El modelo sin las restricciones de `quitar`."""
    return _modelo_de((p, c, n) for p, c, n in _iter_restricciones(modelo)
                      if (p, c) not in quitar)


def alcance_restriccion(clave, regla: Dict) -> str:
    """
    local      : la restriccion solo involucra actividades de phi
    vecindario : involucra al menos una actividad de phi y al menos una ajena
    global     : no involucra actividades de phi
    """
    act_c, act_phi = _actividades_restriccion(clave), actividades_regla(regla)
    if act_c <= act_phi:
        return "local"
    if act_c & act_phi:
        return "vecindario"
    return "global"


def compute_plausibilidad(df_sim: pd.DataFrame, modelo_filtrado: Dict,
                          regla: Dict) -> Dict[str, float]:
    """
    Conformidad de las trazas generadas con el modelo ya filtrado (sin las
    restricciones que phi contradice):

    plaus_media           : % medio de restricciones cumplidas por traza (como CONF)
    plaus_estricta        : % de trazas sin ninguna violacion
    plaus_estricta_<alc>  : lo mismo restringido a las restricciones de cada
                            alcance (local, vecindario, global); NaN si el
                            alcance no tiene restricciones
    """
    tr = list(trazas(df_sim).values())
    restr = list(_iter_restricciones(modelo_filtrado))
    if not tr or not restr:
        return {"plaus_media": float("nan"), "plaus_estricta": float("nan")}

    resultados = _apply_list([list(t) for t in tr], modelo_filtrado)
    alcance = {(p, c): alcance_restriccion(c, regla) for p, c, _ in restr}
    out = {
        "plaus_media": float(np.mean([r["dev_fitness"] for r in resultados])),
        "plaus_estricta": float(np.mean([not r["deviations"] for r in resultados])),
        "n_restricciones": len(restr),
    }
    for alc in ("local", "vecindario", "global"):
        if alc not in alcance.values():
            out[f"plaus_estricta_{alc}"] = float("nan")
            continue
        sin_violar = [
            not any(alcance.get((d[0], d[1])) == alc for d in r["deviations"])
            for r in resultados
        ]
        out[f"plaus_estricta_{alc}"] = float(np.mean(sin_violar))
    return out


def tasa_violacion_por_restriccion(df: pd.DataFrame, modelo: Dict) -> pd.DataFrame:
    """
    Para cada restriccion del modelo, % de trazas de df que la violan.
    Sirve en el hold-out: las restricciones marcadas como contradichas deberian
    ser las que violan las trazas ocultas.
    """
    tr = [list(t) for t in trazas(df).values()]
    resultados = _apply_list(tr, modelo) if tr else []
    violaciones = Counter((d[0], d[1]) for r in resultados for d in r["deviations"])
    filas = [{"plantilla": p, "restriccion": c,
              "tasa_violacion": violaciones[(p, c)] / len(tr) if tr else float("nan")}
             for p, c, _ in _iter_restricciones(modelo)]
    return pd.DataFrame(filas)


def compute_compatibilidad_rol_actividad(df_train: pd.DataFrame, df_sim: pd.DataFrame,
                                         col: str = "role") -> float:
    """
    % de eventos generados cuyo par (rol, actividad) existe en entrenamiento.
    NaN si alguno de los logs no tiene la columna.
    """
    if col not in df_train.columns or col not in df_sim.columns or df_sim.empty:
        return float("nan")
    vistos = set(zip(df_train[col].astype(str), df_train["task"].astype(str)))
    pares = list(zip(df_sim[col].astype(str), df_sim["task"].astype(str)))
    return float(np.mean([p in vistos for p in pares]))


# ─────────────────────────────────────────────────────────────────────────────
# Diversidad
# ─────────────────────────────────────────────────────────────────────────────

def _damerau_levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Distancia de Damerau-Levenshtein (variante de alineamiento optimo):
    insertar, borrar, reemplazar o intercambiar dos actividades vecinas."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costo = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + costo)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


def compute_diversidad(df: pd.DataFrame, max_pares: int = 20000,
                       semilla: int = 42) -> Dict[str, float]:
    """
    diversidad_dl      : distancia de Damerau-Levenshtein promedio entre pares de
                         trazas, normalizada por el largo de la traza mas larga del
                         par. Si hay mas de max_pares pares, se muestrean al azar.
    razon_variantes    : variantes distintas / trazas
    Se reporta para comparar con la diversidad del log real: la meta es
    parecerse, no maximizarla.
    """
    tr = list(trazas(df).values())
    n = len(tr)
    if n < 2:
        return {"diversidad_dl": float("nan"), "razon_variantes": float("nan")}
    total = n * (n - 1) // 2
    if total <= max_pares:
        pares = itertools.combinations(range(n), 2)
    else:
        rng = random.Random(semilla)
        pares = []
        while len(pares) < max_pares:
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                pares.append((i, j))
    dist = [_damerau_levenshtein(tr[i], tr[j]) / max(len(tr[i]), len(tr[j]), 1)
            for i, j in pares]
    return {"diversidad_dl": float(np.mean(dist)), "razon_variantes": len(set(tr)) / n}


# ─────────────────────────────────────────────────────────────────────────────
# Novedad valida
# ─────────────────────────────────────────────────────────────────────────────

def compute_novedad_valida(df_train: pd.DataFrame, df_sim: pd.DataFrame,
                           regla: Dict, modelo_filtrado: Dict) -> Dict[str, float]:
    """
    novedad_valida            : % de trazas generadas que son nuevas, cumplen phi
                                y no violan ninguna restriccion del modelo filtrado
    novedad_valida_entre_phi  : lo mismo, entre las trazas que cumplen phi
    """
    vistas = set(trazas(df_train).values())
    tr = list(trazas(df_sim).values())
    if not tr:
        return {"novedad_valida": float("nan"), "novedad_valida_entre_phi": float("nan")}
    if any(True for _ in _iter_restricciones(modelo_filtrado)):
        plausible = [not r["deviations"] for r in _apply_list([list(t) for t in tr], modelo_filtrado)]
    else:
        plausible = [True] * len(tr)
    cumple = [cumple_regla(t, regla) for t in tr]
    valida = [t not in vistas and c and p for t, c, p in zip(tr, cumple, plausible)]
    con_phi = [v for v, c in zip(valida, cumple) if c]
    return {
        "novedad_valida": float(np.mean(valida)),
        "novedad_valida_entre_phi": float(np.mean(con_phi)) if con_phi else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluacion completa
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_cambio(df_train: pd.DataFrame, df_sim: pd.DataFrame, regla: Dict,
                   modelo_train: Dict,
                   df_ref_fidelidad: Optional[pd.DataFrame] = None,
                   df_ref_novedad: Optional[pd.DataFrame] = None,
                   p_pedida: Optional[float] = None,
                   col_pedido: Optional[str] = None, valor_cumplir=None,
                   col_rol: str = "role", max_len_sat: int = 5) -> Dict[str, float]:
    """
    Todas las metricas de este modulo para un log generado bajo phi.

    df_train         : log de entrenamiento (novedad, modelo DECLARE, roles)
    modelo_train     : modelo pm4py minado de df_train (metrics.discover_conf_model)
    df_ref_fidelidad : referencia para la fidelidad en lo no tocado (por defecto df_train)
    df_ref_novedad   : referencia para la novedad validada (p.ej. trazas ocultas)
    """
    contradichas = restricciones_contradichas(modelo_train, regla, max_len=max_len_sat)
    modelo_f = filtrar_modelo(modelo_train, contradichas)
    ref_fid = df_train if df_ref_fidelidad is None else df_ref_fidelidad

    out: Dict[str, float] = {"n_contradichas": len(contradichas)}
    out.update(compute_novedad(df_train, df_sim))
    out["novedad_no_pedida"] = compute_novedad_no_pedida(df_train, df_sim, regla)
    if df_ref_novedad is not None:
        out.update(compute_novedad_validada(df_train, df_sim, df_ref_novedad))
    out.update(compute_control(df_sim, regla, p_pedida, col_pedido, valor_cumplir))
    out.update(compute_plausibilidad(df_sim, modelo_f, regla))
    out.update(compute_novedad_valida(df_train, df_sim, regla, modelo_f))
    out["freq_act"] = compute_freq_actividades(ref_fid, df_sim)
    out.update(compute_fidelidad_no_tocada(ref_fid, df_sim, regla))
    out["compat_rol_act"] = compute_compatibilidad_rol_actividad(df_train, df_sim, col_rol)
    out.update(compute_diversidad(df_sim))
    out.update({f"{k}_ref": v for k, v in compute_diversidad(ref_fid).items()})
    return out
