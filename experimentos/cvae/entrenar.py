"""
Entrena CVAE (Graziosi et al., ICPM 2024) en un log de Dynamics, con la etiqueta de
una regla: 'deviant' si la traza cumple la regla y 'regular' si no.

CVAE no tiene modelos publicados para los logs de Dynamics. Se entrena con su propio
codigo (version del paper, repo_icpm2024, commit c3be4b0) y su mismo procedimiento
(run_pipeline.py): ray tune con una sola configuracion (NUM_SAMPLES = 1), los
hiperparametros de config_example.py (los del paper), hasta 20.000 epocas con su
early stopping (paciencia 100, min_delta 1, solo con el peso KL en 1) y el mismo
criterio para elegir el checkpoint (menor 'loss' con w_kl = 1, entre los checkpoints
guardados cada 50 epocas). El codigo de CVAE no se modifica.

Datos (formato de los datasets de CVAE, separador ';'):
  - Division comun: TRAIN = subtrain, VAL = validacion (early stopping), TEST = test.
    Entre TRAIN y VAL son los mismos 433 casos de train que usan Dynamics y RULE-GAN.
  - Orden de los eventos: por start_timestamp, con los empates en el orden del log
    comun (el mismo con que se evalua la regla en las metricas).
  - Timestamp: start_timestamp (CVAE usa un solo timestamp por evento y no genera
    duraciones). relative_timestamp_from_start y relative_timestamp_from_previous_activity
    se calculan con las funciones de su scripts/preprocess_log.py, sobre el log completo.
  - Etiqueta: evaluacion.metricas_cambio.cumple_regla sobre esa secuencia.
  - Sin atributos de traza propios (RunningExample no tiene); queda solo
    relative_timestamp_from_start, que CVAE siempre usa.
  - Largo maximo: el del log (MAX_TRACE_LENGTH = None en su config).

Uso (entorno conda cvae):
    python experimentos/cvae/entrenar.py --log RunningExample --regla r1_directly_TaskB_TaskC

Resultado en experimentos/cvae/modelos/<log>/<regla>/:
    modelo.pt          checkpoint elegido (model_state_dict)
    dataset_info.json  actividades, largo maximo y atributos (para generar.py)
    entrenamiento.json epoca elegida, perdidas, epocas corridas, tiempo
    raytune/           progreso del entrenamiento (sin los checkpoints descartados)
Datos en experimentos/cvae/datos/<log>/<regla>/.
"""
import argparse
import datetime
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
REPO = AQUI / "repo_icpm2024"
COMUN = RAIZ / "experimentos" / "comun"
COMMIT = "c3be4b0"

# config_example.py de la version del paper
MAX_NUM_EPOCHS = 20000
CONFIG = {
    "NUM_KL_ANNEALING_CYCLES": MAX_NUM_EPOCHS // 2500,
    "EARLY_STOPPING_PATIENCE": 100,
    "EARLY_STOPPING_MIN_DELTA": 1,
    "EARLY_STOPPING_DEBUG": False,
    "IS_AUTOREGRESSIVE": True,
    "NUM_LSTM_LAYERS": 1,
    "ATTR_E_DIM": 5,
    "ACT_E_DIM": 5,
    "CF_DIM": 200,
    "Z_DIM": 10,
    "DROPOUT_P": 0.05,
    "LR": 3e-4,
    "BATCH_SIZE": 256,
}
CHECKPOINT_CADA = 50


def _cumple(regla: dict):
    sys.path.append(str(RAIZ))
    from evaluacion.metricas_cambio import cumple_regla
    return lambda tareas: cumple_regla(tareas, regla)


def preparar_datos(log: str, regla_id: str) -> Path:
    import pandas as pd
    sys.path.append(str(RAIZ))
    from support_modules import traces_evaluation as te
    sys.path.insert(0, str(REPO))
    from scripts.preprocess_log import (add_relative_timestamp_between_activities,
                                        add_trace_attr_relative_timestamp_to_first_activity)

    regla = te.extract_rules(path=str(COMUN / log / "reglas" / f"{regla_id}.ini"))
    cumple = _cumple(regla)
    df = pd.read_csv(COMUN / log / "log.csv")
    df["caseid"] = df["caseid"].astype(str)
    df["time:timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True, format="ISO8601").dt.tz_convert(None)
    orden = {c: i for i, c in enumerate(dict.fromkeys(df["caseid"]))}
    df["_caso"] = df["caseid"].map(orden)
    df = df.sort_values(["_caso", "time:timestamp"], kind="mergesort").reset_index(drop=True)
    etiqueta = {c: ("deviant" if cumple(g["task"].tolist()) else "regular")
                for c, g in df.groupby("caseid", sort=False)}
    out = pd.DataFrame({"Case ID": df["caseid"], "Activity": df["task"],
                        "time:timestamp": df["time:timestamp"],
                        "label": df["caseid"].map(etiqueta)})
    out = add_trace_attr_relative_timestamp_to_first_activity(
        out, trace_key="Case ID", timestamp_key="time:timestamp",
        custom_timestamp_key="relative_timestamp_from_start")
    out = add_relative_timestamp_between_activities(
        out, trace_key="Case ID", timestamp_key="time:timestamp",
        custom_timestamp_key="relative_timestamp_from_previous_activity")

    destino = AQUI / "datos" / log / regla_id
    destino.mkdir(parents=True, exist_ok=True)
    out.to_csv(destino / "datos.csv", sep=";", index=False)
    resumen = {}
    for nombre, conjunto in (("TRAIN", "subtrain"), ("VAL", "val"), ("TEST", "test")):
        ids = set((COMUN / log / "split" / f"{conjunto}_caseids.txt").read_text().split())
        parte = out[out["Case ID"].isin(ids)]
        parte.to_csv(destino / f"datos_{nombre}.csv", sep=";", index=False)
        casos = parte.groupby("Case ID")["label"].first()
        resumen[nombre] = {"casos": int(len(casos)), "deviant": int((casos == "deviant").sum())}
        if casos.nunique() < 2:
            sys.exit(f"[error] {nombre} tiene una sola etiqueta; CVAE necesita las dos.")
    print(f"[ok] Datos CVAE {log} / {regla_id}: {resumen} -> {destino}")
    return destino


def dataset_info(carpeta_datos: Path) -> dict:
    """DATASET_INFO como lo arma run_pipeline.py (MAX_TRACE_LENGTH = None)."""
    from dataset import GenericDataset
    from utils import get_dataset_attributes_info
    info = {
        "CLASS": GenericDataset,
        "TRAIN": str(carpeta_datos / "datos_TRAIN.csv"),
        "VAL": str(carpeta_datos / "datos_VAL.csv"),
        "TEST": str(carpeta_datos / "datos_TEST.csv"),
        "FULL": str(carpeta_datos / "datos.csv"),
        "CSV_SEPARATOR": ";", "NUM_LABELS": 2,
        "TRACE_ATTRIBUTE_KEYS": ["relative_timestamp_from_start"],
        "ACTIVITY_KEY": "Activity",
        "TIMESTAMP_KEY": "relative_timestamp_from_previous_activity",
        "TRACE_KEY": "Case ID", "LABEL_KEY": "label",
    }
    a = get_dataset_attributes_info(info["FULL"], activity_key=info["ACTIVITY_KEY"],
                                    trace_key=info["TRACE_KEY"],
                                    trace_attributes=info["TRACE_ATTRIBUTE_KEYS"])
    info.update({
        "MAX_TRACE_LENGTH": a["max_trace_length"] + 1,
        "ACTIVITIES": a["activities"],
        "NUM_ACTIVITIES": len(a["activities"]) + 1,
        "TRACE_ATTRIBUTES": a["trace_attributes"],
    })
    return info


def entrenar(log: str, regla_id: str) -> Path:
    import torch
    from functools import partial
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.air.config import RunConfig
    from train import train
    from early_stopper import EarlyStopper

    carpeta_datos = preparar_datos(log, regla_id)
    info = dataset_info(carpeta_datos)
    salida = AQUI / "modelos" / log / regla_id
    if salida.exists():
        shutil.rmtree(salida)
    salida.mkdir(parents=True)
    tmp = salida / "tmp"
    tmp.mkdir()

    config = {**CONFIG, "TMP_PATH": str(tmp), "C_DIM": info["NUM_LABELS"],
              "NUM_EPOCHS": MAX_NUM_EPOCHS}
    trainable = partial(train, dataset_info=info, checkpoint_every=CHECKPOINT_CADA,
                        tmp_path=config["TMP_PATH"], device=torch.device("cpu"))
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": 1}),
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="loss", mode="min", num_samples=1,
            scheduler=ASHAScheduler(max_t=MAX_NUM_EPOCHS, grace_period=50, reduction_factor=2),
            trial_dirname_creator=lambda t: "ensayo"),   # nombre corto (rutas de Windows)
        run_config=RunConfig(
            storage_path=str(salida / "raytune"), name="cvae",
            stop=EarlyStopper(patience=config["EARLY_STOPPING_PATIENCE"],
                              min_delta=config["EARLY_STOPPING_MIN_DELTA"],
                              debug=config["EARLY_STOPPING_DEBUG"])),
    )
    inicio = time.time()
    results = tuner.fit()
    segundos = round(time.time() - inicio, 1)

    # Mismo criterio que run_pipeline.py: menor 'loss' con w_kl = 1
    mejor = results.get_best_result("loss", "min")
    elegido = None
    for checkpoint, m in mejor.best_checkpoints:
        if math.isclose(m["w_kl"], 1) and (elegido is None or m["loss"] < elegido["loss"]):
            elegido = m
    if elegido is None:
        sys.exit("[error] Ningun checkpoint con w_kl = 1")
    epoca = int(elegido["training_iteration"]) - 1   # train.py guarda en checkpoints/<epoch>
    ckpt = Path(mejor.path) / "checkpoints" / str(epoca) / "checkpoint.pt"
    shutil.copy(ckpt, salida / "modelo.pt")

    progreso = mejor.metrics_dataframe
    info_json = {k: v for k, v in info.items() if k != "CLASS"}
    (salida / "dataset_info.json").write_text(json.dumps(info_json, indent=2, default=str),
                                              encoding="utf-8")
    meta = {
        "enfoque": "cvae", "log": log, "regla": regla_id, "codigo": COMMIT,
        "config": config, "epocas_corridas": int(progreso["training_iteration"].max()),
        "epoca_elegida": epoca + 1, "loss": elegido["loss"], "val_loss": elegido["val_loss"],
        "segundos": segundos, "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (salida / "entrenamiento.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Los checkpoints descartados no se guardan (uno cada 50 epocas)
    shutil.rmtree(Path(mejor.path) / "checkpoints", ignore_errors=True)
    for p in Path(mejor.path).glob("checkpoint_*"):
        shutil.rmtree(p, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ok] CVAE {log} / {regla_id}: epoca {epoca + 1} de {meta['epocas_corridas']}, "
          f"loss {elegido['loss']:.3f}, {segundos:.0f} s -> {salida}")
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--regla", required=True)
    ap.add_argument("--solo-datos", action="store_true", help="solo prepara los datos")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.chdir(REPO)                    # el codigo de CVAE usa rutas e imports relativos a su repo
    sys.path.insert(0, str(REPO))
    if args.solo_datos:
        preparar_datos(args.log, args.regla)
    else:
        entrenar(args.log, args.regla)
