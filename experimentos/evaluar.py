"""
Evalua los logs generados por cada enfoque bajo una regla (escenario restrictivo),
con las dimensiones de la metodologia: eficiencia, fidelidad, novedad, control
condicional, plausibilidad y diversidad.

Referencias (division comun, experimentos/comun/<log>/):
  - train            : novedad (que es nuevo), modelo DECLARE para la plausibilidad
                       (MINERful, soporte >= 0.9) y pares rol-actividad.
  - test que cumple  : fidelidad. Es el comportamiento real bajo la regla, que el
                       generador no vio.
La regla se evalua con evaluacion.metricas_cambio.cumple_regla sobre la secuencia de
actividades ordenada por start_timestamp (los empates quedan en el orden del archivo).

Filas:
  - un enfoque por carpeta de ejecuciones (Dynamics, RULE-GAN, CVAE, CVAE con rechazo),
    con una fila por corrida y el resumen (media y desviacion) por enfoque;
  - 'Train filtrado': las trazas del train que cumplen la regla. Es la solucion
    trivial para un cambio restrictivo y sirve de cota.

Uso:
    python experimentos/evaluar.py --log RunningExample --regla r1_directly_TaskB_TaskC

Resultado en experimentos/resultados/<log>/<regla>/:
    por_corrida.csv, resumen.csv, variantes.csv
"""
import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
COMUN = AQUI / "comun"

ENFOQUES = {   # nombre: (carpeta del enfoque, sufijo de la carpeta de la regla)
    "Dynamics": ("dynamics", ""),
    "RULE-GAN": ("rule_gan", ""),
    "CVAE": ("cvae", ""),
    "CVAE + rechazo": ("cvae", "_rechazo"),
}

# Metricas por dimension (para ordenar el resumen)
DIMENSIONES = {
    "Eficiencia": ["seg_entrenamiento", "seg_generacion", "intentos_por_traza"],
    "Fidelidad": ["RED", "CTD_horas", "2GD", "freq_act", "2GD_no_tocada", "freq_act_no_tocada"],
    "Novedad": ["novedad_trazas", "novedad_variantes", "novedad_no_pedida"],
    "Control condicional": ["prop_cumple"],
    "Plausibilidad": ["CONF", "plaus_estricta", "plaus_estricta_vecindario",
                      "plaus_estricta_global", "compat_rol_act", "novedad_valida"],
    "Diversidad": ["diversidad_dl", "diversidad_dl_ref", "razon_variantes", "razon_variantes_ref"],
}


def _log(ruta: Path):
    from evaluacion.metrics import load_log
    df = load_log(str(ruta))
    df["caseid"] = df["caseid"].astype(str)
    if "end_timestamp" not in df.columns:
        df["end_timestamp"] = df["start_timestamp"]
    # orden estable por inicio: los empates quedan en el orden del archivo
    orden = {c: i for i, c in enumerate(dict.fromkeys(df["caseid"]))}
    df = (df.assign(_c=df["caseid"].map(orden))
            .sort_values(["_c", "start_timestamp"], kind="mergesort")
            .drop(columns="_c").reset_index(drop=True))
    return df


def _comun(log: str, conjunto: str):
    df = _log(COMUN / log / "log.csv")
    ids = set((COMUN / log / "split" / f"{conjunto}_caseids.txt").read_text().split())
    return df[df["caseid"].isin(ids)].reset_index(drop=True)


def evaluar_log(df_sim, ref: dict, regla: dict) -> dict:
    from evaluacion.metrics import compute_2gd, compute_conf, compute_ctd, compute_red
    from evaluacion.metricas_cambio import evaluar_cambio
    sim = df_sim.copy()
    if "role" not in sim.columns and "user" in sim.columns and sim["user"].notna().any():
        sim["role"] = sim["user"]                    # los generadores guardan el rol en 'user'
    out = evaluar_cambio(ref["train"], sim, regla, ref["modelo"],
                         df_ref_fidelidad=ref["test_phi"], p_pedida=1.0)
    out.update({
        "RED": compute_red(ref["test_phi"], sim),
        "CTD_horas": compute_ctd(ref["test_phi"], sim) / 3600.0,
        "2GD": compute_2gd(ref["test_phi"], sim),
        "CONF": compute_conf(ref["train"], sim, cached_model=ref["modelo"])[0],
    })
    return out


def evaluar(log: str, regla_id: str) -> None:
    import numpy as np
    import pandas as pd
    from evaluacion.metrics import discover_conf_model
    from evaluacion.metricas_cambio import cumple_regla, separar_por_regla, trazas
    from support_modules import traces_evaluation as te

    regla = te.extract_rules(path=str(COMUN / log / "reglas" / f"{regla_id}.ini"))
    train, test = _comun(log, "train"), _comun(log, "test")
    test_phi, _ = separar_por_regla(test, regla)
    train_phi, _ = separar_por_regla(train, regla)
    modelo, restricciones = discover_conf_model(train, 0.9)
    ref = {"train": train, "test_phi": test_phi, "modelo": modelo}
    print(f"[evaluar] {log} / {regla_id}: train {train['caseid'].nunique()} casos, "
          f"test que cumple {test_phi['caseid'].nunique()}, {len(restricciones)} restricciones")

    filas, variantes = [], []
    vistas = set(trazas(train).values())

    def _variantes(nombre, df):
        for t, k in Counter(trazas(df).values()).items():
            variantes.append({"enfoque": nombre, "variante": " > ".join(t), "trazas": k,
                              "cumple": cumple_regla(t, regla), "en_train": t in vistas})

    filas.append({"enfoque": "Train filtrado", "corrida": 0,
                  **evaluar_log(train_phi, ref, regla)})
    _variantes("Train filtrado", train_phi)
    _variantes("Test que cumple (referencia)", test_phi)

    for nombre, (carpeta, sufijo) in ENFOQUES.items():
        base = AQUI / carpeta / "ejecuciones" / log / (regla_id + sufijo)
        corridas = sorted(base.glob("corrida_*")) if base.exists() else []
        if not corridas:
            print(f"[evaluar] sin corridas: {nombre} ({base})")
            continue
        for c in corridas:
            meta = json.loads((c / "meta.json").read_text(encoding="utf-8"))
            df = _log(c / f"{log}.csv")
            n = df["caseid"].nunique()
            seg_ent = meta.get("segundos_entrenamiento")
            filas.append({
                "enfoque": nombre, "corrida": int(c.name.split("_")[1]),
                "n_trazas": n, "seg_entrenamiento": seg_ent,
                "seg_generacion": meta.get("segundos"),
                "intentos_por_traza": meta["intentos"] / n if meta.get("intentos") and n else np.nan,
                **evaluar_log(df, ref, regla),
            })
            _variantes(nombre, df)
            print(f"[evaluar] {nombre} {c.name}: ok")

    por_corrida = pd.DataFrame(filas)
    salida = AQUI / "resultados" / log / regla_id
    salida.mkdir(parents=True, exist_ok=True)
    por_corrida.to_csv(salida / "por_corrida.csv", index=False)

    columnas = [m for ms in DIMENSIONES.values() for m in ms if m in por_corrida.columns]
    g = por_corrida.groupby("enfoque", sort=False)[columnas]
    media, desv = g.mean(), g.std(ddof=1)
    filas_res = []
    for dim, ms in DIMENSIONES.items():
        for m in ms:
            if m not in por_corrida.columns:
                continue
            fila = {"dimension": dim, "metrica": m}
            for enf in media.index:
                mu, sd = media.loc[enf, m], desv.loc[enf, m]
                fila[enf] = (f"{mu:.3f}" if pd.isna(sd) else f"{mu:.3f} ± {sd:.3f}") if pd.notna(mu) else ""
            filas_res.append(fila)
    resumen = pd.DataFrame(filas_res)
    resumen.to_csv(salida / "resumen.csv", index=False)
    (pd.DataFrame(variantes).groupby(["enfoque", "variante", "cumple", "en_train"], sort=False)
       ["trazas"].sum().reset_index()
       .sort_values(["enfoque", "trazas"], ascending=[True, False])
       .to_csv(salida / "variantes.csv", index=False))
    pd.set_option("display.width", 250)
    print(resumen.to_string(index=False))
    print(f"[ok] {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--regla", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(RAIZ)
    sys.path.insert(0, str(RAIZ))
    evaluar(args.log, args.regla)
