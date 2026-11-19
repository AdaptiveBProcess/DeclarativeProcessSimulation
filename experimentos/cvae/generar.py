"""
Genera logs con los modelos publicados de CVAE (Graziosi et al., ICPM 2024), sin
reentrenar, usando su propio codigo (evaluation/generate_log.py).

Se usa la version del codigo del paper de ICPM 2024 (commit c3be4b0, anterior a
"Add resource generation to the CVAE", 4ea4553): los modelos de
additional_material/models fueron entrenados con ella y no generan recursos. El
codigo actual del repositorio (con recursos) no puede cargarlos.

Requisitos:
  - entorno conda cvae (ver experimentos/cvae/README.md)
  - copia de trabajo en experimentos/cvae/repo_icpm2024

Uso:
    python experimentos/cvae/generar.py --dataset bpic2012_a --corridas 3
    python experimentos/cvae/generar.py --dataset bpic2012_a --proporcion 0.165

  --proporcion p : fraccion de trazas generadas con la etiqueta 'deviant'
                   (= la condicion se cumple). Por defecto 1.0, como Dynamics y
                   RULE-GAN con variation = =1.
  --n N          : trazas a generar (por defecto, las trazas del TEST de CVAE).

Cada corrida queda en experimentos/cvae/ejecuciones/<dataset>/p<proporcion>/corrida_XX/:
    <dataset>.csv        formato comun: caseid, task, user (vacio: CVAE no genera
                         recursos), start_timestamp, end_timestamp, label
                         (un solo timestamp por evento -> start = end)
    salida_cvae.csv/.xes salida original de CVAE
    meta.json
"""
import argparse
import datetime
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI / "repo_icpm2024"
COMMIT = "c3be4b0"

# Valores de config_example.py de esa version (los del paper)
DATASETS = {
    "bpic2012_a":    {"max_len": 100, "attrs": ["AMOUNT_REQ"],             "modelo": "bpic2012_a.pkl"},
    "bpic2012_b":    {"max_len": 100, "attrs": ["AMOUNT_REQ"],             "modelo": "bpic2012_b.pt"},
    "traffic_fines": {"max_len": 20,  "attrs": ["vehicleClass", "amount"], "modelo": "traffic_fines.pt"},
    "sepsis":        {"max_len": 50,  "attrs": ["Diagnose", "Age"],        "modelo": "sepsis.pt"},
}
HIPERPARAMETROS = {"NUM_LSTM_LAYERS": 1, "ATTR_E_DIM": 5, "ACT_E_DIM": 5,
                   "CF_DIM": 200, "Z_DIM": 10, "DROPOUT_P": 0.05}


class _MetadatosRay:
    """Reemplazo de ray.air.checkpoint._CheckpointMetadata (bpic2012_a.pkl se guardo
    con una version vieja de ray que ya no lo trae). Solo son metadatos del
    checkpoint; los pesos no dependen de ray."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __setstate__(self, state):
        self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "ray.air.checkpoint":
            return _MetadatosRay
        return super().find_class(module, name)


def _dataset_info(nombre: str) -> dict:
    from dataset import GenericDataset
    from utils import get_dataset_attributes_info
    d = DATASETS[nombre]
    base = REPO / "datasets" / nombre
    info = {
        "CLASS": GenericDataset,
        "TRAIN": str(base / f"{nombre}_TRAIN.csv"), "VAL": str(base / f"{nombre}_VAL.csv"),
        "TEST": str(base / f"{nombre}_TEST.csv"), "FULL": str(base / f"{nombre}.csv"),
        "CSV_SEPARATOR": ";", "NUM_LABELS": 2,
        "TRACE_ATTRIBUTE_KEYS": ["relative_timestamp_from_start"] + d["attrs"],
        "ACTIVITY_KEY": "Activity", "TIMESTAMP_KEY": "relative_timestamp_from_previous_activity",
        "TRACE_KEY": "Case ID", "LABEL_KEY": "label",
    }
    a = get_dataset_attributes_info(info["FULL"], activity_key=info["ACTIVITY_KEY"],
                                    trace_key=info["TRACE_KEY"],
                                    trace_attributes=info["TRACE_ATTRIBUTE_KEYS"])
    info.update({
        "MAX_TRACE_LENGTH": d["max_len"] + 1,
        "ACTIVITIES": a["activities"], "NUM_ACTIVITIES": len(a["activities"]) + 1,
        "TRACE_ATTRIBUTES": a["trace_attributes"],
    })
    return info


def _modelo(nombre: str, info: dict):
    import torch
    from model import VAE
    ruta = REPO / "additional_material" / "models" / DATASETS[nombre]["modelo"]
    if ruta.suffix == ".pkl":
        with open(ruta, "rb") as f:
            checkpoint = _Unpickler(f).load()
    else:
        checkpoint = torch.load(ruta, map_location="cpu")
    h = HIPERPARAMETROS
    model = VAE(trace_attributes=info["TRACE_ATTRIBUTES"], num_activities=info["NUM_ACTIVITIES"],
                max_trace_length=info["MAX_TRACE_LENGTH"], num_lstm_layers=h["NUM_LSTM_LAYERS"],
                attr_e_dim=h["ATTR_E_DIM"], act_e_dim=h["ACT_E_DIM"], cf_dim=h["CF_DIM"],
                c_dim=info["NUM_LABELS"], z_dim=h["Z_DIM"], dropout_p=h["DROPOUT_P"]).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, ruta.name


def _formato_comun(salida_csv: Path, destino_csv: Path) -> dict:
    import pandas as pd
    df = pd.read_csv(salida_csv, sep=";")
    out = pd.DataFrame({
        "caseid": df["case:concept:name"].astype(str),
        "task": df["concept:name"],
        "user": df["org:resource"] if "org:resource" in df.columns else None,
        "start_timestamp": df["time:timestamp"],
        "end_timestamp": df["time:timestamp"],
        "label": df["case:label"],
    })
    out.to_csv(destino_csv, index=False)
    return {"n_trazas": int(out["caseid"].nunique()), "n_eventos": int(len(out))}


def generar(nombre: str, corridas: int, semilla: int, proporcion: float, n: int = None) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from evaluation.generate_log import generate_log

    info = _dataset_info(nombre)
    model, archivo_modelo = _modelo(nombre, info)
    if n is None:
        n = int(pd.read_csv(info["TEST"], sep=";")[info["TRACE_KEY"]].nunique())
    n_dev = int(round(proporcion * n))
    config_gen = {k: v for k, v in {"deviant": n_dev, "regular": n - n_dev}.items() if v > 0}

    for i in range(1, corridas + 1):
        sem = semilla + i - 1
        random.seed(sem)
        np.random.seed(sem)
        torch.manual_seed(sem)
        destino = AQUI / "ejecuciones" / nombre / f"p{proporcion:g}" / f"corrida_{i:02d}"
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir(parents=True)
        inicio = datetime.datetime.now()
        generate_log(model, generation_config=config_gen, dataset_info=info,
                     output_path=str(destino), output_name="salida_cvae")
        meta = {
            "enfoque": "cvae", "dataset": nombre, "corrida": i, "semilla": sem,
            "codigo": COMMIT, "modelo": archivo_modelo, "proporcion_deviant": proporcion,
            "generacion": config_gen,
            "inicio": inicio.isoformat(timespec="seconds"),
            "segundos": round((datetime.datetime.now() - inicio).total_seconds(), 1),
            **_formato_comun(destino / "salida_cvae.csv", destino / f"{nombre}.csv"),
        }
        (destino / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] {nombre} / p={proporcion:g} / corrida {i}: {destino}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--corridas", type=int, default=1)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--proporcion", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.chdir(REPO)                    # el codigo de CVAE usa rutas e imports relativos a su repo
    sys.path.insert(0, str(REPO))
    generar(args.dataset, args.corridas, args.semilla, args.proporcion, args.n)
