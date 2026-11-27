"""
Evaluacion de modelos RULE-GAN contra la VALIDACION, para elegir configuracion,
epoca (checkpoint) y temperatura de muestreo sin usar el test.

Se evalua el generador base (sin regla): el comportamiento generado debe parecerse
al de la validacion. La regla (opcion d) se ajusta despues, sobre el modelo elegido.

Metricas (todas contra la validacion; menor es mejor):
    RED, CTD, 2GD          evaluacion/metrics.py
    freq_act               distancia entre frecuencias de actividades
    1-CONF                 restricciones DECLARE minadas del SUBTRAIN (soporte >= 0.9)
    div_gap                |diversidad Damerau-Levenshtein generada - de la validacion|
Ademas se reporta pct_vacias (trazas vacias descartadas), que no entra al objetivo.

Objetivo escalar (para la busqueda): promedio de cada metrica dividida por la de V4
(primer ensayo de la busqueda, ultima epoca, temperatura 1). V4 vale 1; menor es mejor.

Temperatura: las probabilidades de actividad y de rol de cada posicion se elevan a
1/T y se renormalizan antes de muestrear (T < 1: mas conservador; T > 1: mas diverso).
La grilla es {0.5, 0.6, 0.7, 0.85, 1, 1.15, 1.3}; 0.5 y 0.6 se agregaron porque en la
busqueda amplia de RE se eligio 0.7 (el minimo de entonces) en 28 de 30 configuraciones.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
COMUN = RAIZ / "experimentos" / "comun"
METRICAS = ["RED", "CTD", "2GD", "freq_act", "1-CONF", "div_gap"]
TEMPERATURAS = (0.5, 0.6, 0.7, 0.85, 1.15, 1.3)   # ademas de T = 1


# ── Datos de referencia ──────────────────────────────────────────────────────

def _log_comun(log: str, conjunto: str) -> pd.DataFrame:
    df = pd.read_csv(COMUN / log / "log.csv")
    ids = set((COMUN / log / "split" / f"{conjunto}_caseids.txt").read_text().split())
    df = df[df["caseid"].astype(str).isin(ids)].copy()
    df["caseid"] = df["caseid"].astype(str)
    for col in ("start_timestamp", "end_timestamp"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(None)
    return df


def referencia(log: str, cache: Path) -> dict:
    """Validacion, inicio de la validacion, modelo DECLARE del subtrain y diversidad
    de la validacion. El modelo DECLARE (MINERful) se guarda en cache."""
    from evaluacion.metrics import discover_conf_model
    from evaluacion.metricas_cambio import compute_diversidad
    val = _log_comun(log, "val")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        modelo_conf = pickle.loads(cache.read_bytes())
    else:
        modelo_conf, _ = discover_conf_model(_log_comun(log, "subtrain"), 0.9)
        cache.write_bytes(pickle.dumps(modelo_conf))
    return {
        "val": val,
        "n_val": int(val["caseid"].nunique()),
        "inicio": val["start_timestamp"].min(),
        "modelo_conf": modelo_conf,
        "div_val": compute_diversidad(val)["diversidad_dl"],
    }


# ── Generador ────────────────────────────────────────────────────────────────

def parametros_modelo(carpeta: Path) -> dict:
    return json.loads((carpeta / "parameters" / "model_parameters.json").read_text(encoding="utf-8"))


def cargar_generador(carpeta: Path, epoca: int):
    """Reconstruye el generador con la arquitectura del modelo y carga los pesos del
    checkpoint epoca_XXXX.h5."""
    from GenerativeGAN.model_training.models.model_wgan_transformer import build_transformer_generator
    mp = parametros_modelo(carpeta)
    gen = build_transformer_generator(
        int(mp["latent_dim"]), int(mp["max_trace_size"]), len(mp["index_ac"]), len(mp["index_rl"]),
        d_model=int(mp["d_model"]), num_heads=int(mp["num_heads"]),
        num_blocks=int(mp["num_blocks"]), ff_dim=int(mp["ff_dim"]), dropout=float(mp["dropout"]))
    gen.load_weights(str(carpeta / "checkpoints" / f"epoca_{epoca:04d}.h5"))
    return gen, mp


def epocas_disponibles(carpeta: Path) -> list:
    return sorted(int(p.stem.split("_")[1]) for p in (carpeta / "checkpoints").glob("epoca_*.h5"))


def _con_temperatura(p: np.ndarray, t: float) -> np.ndarray:
    p = np.clip(p, 0, None).astype(np.float64)
    if p.sum() <= 0:
        return np.ones_like(p) / len(p)
    if t != 1.0:
        p = np.power(p, 1.0 / t)
    return p / p.sum()


def generar_libre(gen, mp: dict, n: int, temperatura: float, semilla: int, inicio,
                  lote: int = 64, presupuesto: int = 10, acepta=None, contadores: dict = None):
    """n trazas. Misma decodificacion que GANPredictor._decode_trace (corta en 'end',
    salta 'start', reescala dur/wait), con temperatura.

    Sin `acepta` (seleccion) se guarda toda traza no vacia. Con `acepta` (generacion
    bajo una regla, generar.py) se guarda solo si acepta(lista de actividades) es True;
    el presupuesto (presupuesto * n trazas generadas) es el mismo en ambos casos.
    Si se pasa `contadores`, se llena con intentos, vacias, rechazadas y aceptadas.
    Devuelve (DataFrame, % de trazas vacias descartadas)."""
    from GenerativeGAN.model_prediction.gan_predictor import GANPredictor
    rng = np.random.default_rng(semilla)
    index_ac = {int(k): v for k, v in mp["index_ac"].items()}
    index_rl = {int(k): v for k, v in mp["index_rl"].items()}
    n_ac, n_rl = len(index_ac), len(index_rl)
    filas, aceptadas, vacias, intentos, rechazadas = [], 0, 0, 0, 0
    while aceptadas < n and intentos < presupuesto * n:
        ruido = rng.normal(0, 1, (lote, int(mp["latent_dim"]))).astype(np.float32)
        salida = gen.predict(ruido, verbose=0)
        for paso_traza in salida:
            if aceptadas >= n:
                break
            intentos += 1
            traza, ts = [], inicio
            for paso in paso_traza:
                ac = int(rng.choice(n_ac, p=_con_temperatura(paso[:n_ac], temperatura)))
                rl = int(rng.choice(n_rl, p=_con_temperatura(paso[n_ac:n_ac + n_rl], temperatura)))
                tarea, rol = index_ac.get(ac), index_rl.get(rl, "UNKNOWN")
                if tarea in (None, "end"):
                    break
                if tarea == "start":
                    continue
                if rol in ("start", "end"):
                    rol = "UNKNOWN"
                dur = GANPredictor._rescale(float(paso[-2]), mp["scale_args"].get("dur", {}), mp["norm_method"])
                wait = GANPredictor._rescale(float(paso[-1]), mp["scale_args"].get("wait", {}), mp["norm_method"])
                dur = 0.0 if not np.isfinite(dur) else max(dur, 0.0)
                wait = 0.0 if not np.isfinite(wait) else max(wait, 0.0)
                inicio_ev = ts + pd.Timedelta(seconds=wait)
                fin_ev = inicio_ev + pd.Timedelta(seconds=dur)
                traza.append({"task": tarea, "user": rol,
                              "start_timestamp": inicio_ev, "end_timestamp": fin_ev})
                ts = fin_ev
            if not traza:
                vacias += 1
                continue
            if acepta is not None and not acepta([e["task"] for e in traza]):
                rechazadas += 1
                continue
            caso = f"GEN{aceptadas:05d}"
            filas.extend({"caseid": caso, **e} for e in traza)
            aceptadas += 1
    if contadores is not None:
        contadores.update(intentos=intentos, vacias=vacias, rechazadas=rechazadas,
                          aceptadas=aceptadas)
    df = pd.DataFrame(filas, columns=["caseid", "task", "user", "start_timestamp", "end_timestamp"])
    return df, 100.0 * vacias / max(intentos, 1)


# ── Metricas y objetivo ──────────────────────────────────────────────────────

def metricas(df_gen: pd.DataFrame, ref: dict) -> dict:
    from evaluacion.metrics import compute_red, compute_ctd, compute_2gd, compute_conf
    from evaluacion.metricas_cambio import compute_freq_actividades, compute_diversidad
    val = ref["val"]
    conf, _ = compute_conf(val, df_gen, cached_model=ref["modelo_conf"])
    return {
        "RED": compute_red(val, df_gen),
        "CTD": compute_ctd(val, df_gen),
        "2GD": compute_2gd(val, df_gen),
        "freq_act": compute_freq_actividades(val, df_gen),
        "1-CONF": 1.0 - conf,
        "div_gap": abs(compute_diversidad(df_gen)["diversidad_dl"] - ref["div_val"]),
    }


def objetivo(valores: dict, ref_v4: dict) -> float:
    """Promedio de cada metrica relativa a V4 (V4 = 1, menor es mejor)."""
    return float(np.mean([valores[m] / max(ref_v4[m], 1e-9) for m in METRICAS]))


def evaluar_configuracion(gen, mp: dict, ref: dict, temperatura: float, n_gen: int,
                          semilla: int) -> dict:
    """Promedio de las metricas sobre n_gen logs generados (cada uno del tamano de la
    validacion), como en Camargo et al. (5 logs por modelo)."""
    filas = []
    for g in range(n_gen):
        df, vacias = generar_libre(gen, mp, ref["n_val"], temperatura, semilla + g, ref["inicio"])
        filas.append({**metricas(df, ref), "pct_vacias": vacias})
    media = pd.DataFrame(filas).mean().to_dict()
    desv = pd.DataFrame(filas).std(ddof=0).add_suffix("_std").to_dict()
    return {**media, **desv}


def evaluar_temperaturas(carpeta: Path, epoca: int, ref: dict, ref_v4: dict, temperaturas,
                         n_gen: int = 5, semilla: int = 0) -> list:
    """Evalua el checkpoint `epoca` con cada temperatura. Devuelve una fila por
    temperatura, con el objetivo relativo a V4."""
    gen, mp = cargar_generador(carpeta, epoca)
    filas = []
    for t in temperaturas:
        f = {"epoca": epoca, "temperatura": t,
             **evaluar_configuracion(gen, mp, ref, t, n_gen, semilla)}
        f["objetivo"] = objetivo(f, ref_v4)
        filas.append(f)
    return filas


def evaluar_modelo(carpeta: Path, ref: dict, ref_v4: dict = None, n_gen: int = 5,
                   temperaturas=TEMPERATURAS, semilla: int = 0):
    """Elige la epoca con T = 1 y luego la temperatura en esa epoca.

    Si ref_v4 es None (primer ensayo = V4), la referencia se toma de este modelo en
    su ultima epoca con T = 1 y se devuelve para normalizar los ensayos siguientes.
    Devuelve (filas evaluadas, mejor fila, ref_v4)."""
    filas = []
    epocas = epocas_disponibles(carpeta)
    for epoca in epocas:
        gen, mp = cargar_generador(carpeta, epoca)
        filas.append({"epoca": epoca, "temperatura": 1.0,
                      **evaluar_configuracion(gen, mp, ref, 1.0, n_gen, semilla)})
    if ref_v4 is None:
        ultima = [f for f in filas if f["epoca"] == epocas[-1]][0]
        ref_v4 = {m: ultima[m] for m in METRICAS}
    for f in filas:
        f["objetivo"] = objetivo(f, ref_v4)
    mejor_epoca = min(filas, key=lambda f: f["objetivo"])["epoca"]
    filas += evaluar_temperaturas(carpeta, mejor_epoca, ref, ref_v4, temperaturas, n_gen, semilla)
    mejor = min(filas, key=lambda f: f["objetivo"])
    return filas, mejor, ref_v4
