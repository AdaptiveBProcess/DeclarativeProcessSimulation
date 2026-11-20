"""
Genera logs con un modelo RULE-GAN entrenado bajo una regla.

Usa GANPredictor tal cual (generacion por lotes y rechazo de trazas que no
cumplen, con presupuesto de 10x casos). Para que la comparacion con Dynamics sea
pareja:
  - genera tantas trazas como casos de test comunes (107 en RunningExample);
  - la proporcion pedida sale de la 'variation' de la regla (=1 -> 100% cumple),
    igual que en Dynamics (la ruta GAN de dg_prediction.py usaba la de train);
  - la proporcion previa de la regla se calcula en el train comun para cada regla;
  - la primera traza arranca en el inicio del test, como en Dynamics.

Uso:
    python experimentos/rule_gan/generar.py --log RunningExample \
        --regla r1_directly_TaskB_TaskC --corridas 3 [--modelo <carpeta>]

Cada corrida queda en experimentos/rule_gan/ejecuciones/<log>/<regla>/corrida_XX/:
    <log>.csv            caseid, task, user (= rol generado), start_timestamp, end_timestamp
    regla.ini, meta.json
"""
import argparse
import datetime
import json
import os
import random
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
COMUN = RAIZ / "experimentos" / "comun"


def _modelo(log: str, nombre: str = None) -> Path:
    base = AQUI / "modelos" / log
    if nombre:
        return base / nombre
    carpetas = [p for p in base.iterdir() if (p / "parameters" / "model_parameters.json").exists()]
    if not carpetas:
        sys.exit(f"[error] No hay modelos entrenados en {base}; correr entrenar.py")
    return max(carpetas, key=lambda p: p.name)


def _prior_regla(log: str, regla: dict, entrenamiento: dict):
    """(casos de train que cumplen la regla, casos de train) en el train comun."""
    import pandas as pd
    from support_modules import traces_evaluation as te
    df = pd.read_csv(COMUN / log / "log.csv")
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True)
    train = set((COMUN / log / "split" / "train_caseids.txt").read_text().split())
    train -= set(entrenamiento.get("casos_ocultos") or [])
    df = df[df["caseid"].astype(str).isin(train)]
    cumple = df.groupby("caseid").apply(lambda g: te.case_satisfies_rule(g, regla["path"], regla["rule"]))
    return int(cumple.sum()), int(len(cumple))


def _inicio_test(log: str):
    import pandas as pd
    df = pd.read_csv(COMUN / log / "log.csv")
    test = set((COMUN / log / "split" / "test_caseids.txt").read_text().split())
    df = df[df["caseid"].astype(str).isin(test)]
    return pd.to_datetime(df["start_timestamp"], utc=True).dt.tz_convert(None).min(), len(test)


def generar(log: str, regla_id: str, corridas: int, semilla: int, modelo: str, variante: str) -> None:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from GenerativeGAN.model_prediction.gan_predictor import GANPredictor
    from support_modules import traces_evaluation as te

    carpeta = _modelo(log, modelo)
    mp = json.loads((carpeta / "parameters" / "model_parameters.json").read_text(encoding="utf-8"))
    ent_path = carpeta / "entrenamiento.json"
    entrenamiento = json.loads(ent_path.read_text(encoding="utf-8")) if ent_path.exists() else {}
    ini = COMUN / log / "reglas" / f"{regla_id}.ini"
    regla = te.extract_rules(path=str(ini))
    p_pedida = regla["prop_variation"] if regla.get("variation") == "=" else None
    if p_pedida is None:
        sys.exit("[error] La regla debe fijar la proporcion con 'variation = =p'")
    pos, n = _prior_regla(log, regla, entrenamiento)
    inicio_test, n_test = _inicio_test(log)

    index_ac = {int(k): v for k, v in mp["index_ac"].items()}
    index_rl = {int(k): v for k, v in mp["index_rl"].items()}

    for i in range(1, corridas + 1):
        sem = semilla + i - 1
        random.seed(sem)
        np.random.seed(sem)
        tf.random.set_seed(sem)
        destino = AQUI / "ejecuciones" / log / regla_id / f"corrida_{i:02d}"
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir(parents=True)

        parms = {
            "latent_dim": mp.get("latent_dim", 64),
            "index_ac": index_ac, "index_rl": index_rl,
            "ac_index": {v: k for k, v in index_ac.items()},
            "num_cases": n_test, "len_log": n_test,
            "rules": regla, "new_prop_cases": p_pedida,
            "pos_cases_org": pos, "total_cases_org": n,
            "start_time": inicio_test,
            "scale_args": mp["scale_args"], "norm_method": mp["norm_method"],
            "variant": variante,
        }
        inicio = datetime.datetime.now()
        log_gen = GANPredictor().predict(parms, str(carpeta / mp["model_file"]), None, None, None)
        columnas = ["caseid", "task", "user", "start_timestamp", "end_timestamp"]
        df = pd.DataFrame(log_gen, columns=["caseid", "task", "role", "start_timestamp",
                                            "end_timestamp"]).rename(columns={"role": "user"})
        df = df[~df["task"].isin(["Start", "End", "start", "end"])][columnas]
        if df.empty:
            print(f"[aviso] {regla_id}: ninguna traza generada cumple la regla dentro del "
                  f"presupuesto de intentos (corrida {i}).")
        df.to_csv(destino / f"{log}.csv", index=False)
        shutil.copy(ini, destino / "regla.ini")
        meta = {
            "enfoque": "rule_gan", "log": log, "regla": regla_id, "corrida": i,
            "semilla": sem, "variante": variante, "modelo": carpeta.name,
            "epocas_modelo": entrenamiento.get("epocas"),
            "p_pedida": p_pedida, "prior_train": [pos, n],
            "inicio": inicio.isoformat(timespec="seconds"),
            "segundos": round((datetime.datetime.now() - inicio).total_seconds(), 1),
            "n_trazas": int(df["caseid"].nunique()) if len(df) else 0,
            "n_eventos": int(len(df)),
        }
        (destino / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] {log} / {regla_id} / corrida {i}: {destino}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--regla", required=True)
    ap.add_argument("--corridas", type=int, default=1)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--modelo", default=None, help="carpeta del modelo (por defecto, la mas reciente)")
    ap.add_argument("--variante", default="Rules Based Random Choice")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.chdir(RAIZ)
    sys.path.insert(0, str(RAIZ))
    generar(args.log, args.regla, args.corridas, args.semilla, args.modelo, args.variante)
