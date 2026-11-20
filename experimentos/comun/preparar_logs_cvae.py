"""
Arma el log y la division comunes para los logs de CVAE (BPIC2012 y Traffic Fines),
con la division de sus autores (version ICPM 2024, experimentos/cvae/repo_icpm2024).

    subtrain = <ds>_TRAIN   val = <ds>_VAL   train = TRAIN + VAL
    test     = <ds>_TEST (BPIC2012) / <ds>_TEST_cropped (Traffic Fines: el test del 5%
               que reporta el paper, 6.480 trazas)

BPIC2012 A y B tienen los mismos casos en cada division (solo cambia la etiqueta), asi
que se arma un unico BPIC2012 desde los archivos de bpic2012_a.

Los logs de CVAE tienen un solo timestamp por evento: start_timestamp = end_timestamp
(duraciones cero; RULE-GAN solo aprende las esperas). La etiqueta de CVAE se conserva
en la columna 'label'.

Resultado, por log:
    experimentos/comun/<log>/log.csv   caseid, task, user, start_timestamp, end_timestamp, label
    experimentos/comun/<log>/split/{subtrain,val,train,test}_caseids.txt

Uso:
    python experimentos/comun/preparar_logs_cvae.py
"""
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "cvae" / "repo_icpm2024" / "datasets"
LOGS = {"BPIC2012": ("bpic2012_a", "TEST"), "TrafficFines": ("traffic_fines", "TEST_cropped")}
COLS = ["Case ID", "Activity", "Resource", "time:timestamp", "label"]


def _leer(ds: str, parte: str) -> pd.DataFrame:
    df = pd.read_csv(DATOS / ds / f"{ds}_{parte}.csv", sep=";", usecols=lambda c: c in COLS,
                     dtype={"Case ID": str, "Resource": str})
    df["parte"] = parte
    return df


def preparar(log: str, ds: str, parte_test: str) -> None:
    partes = {"subtrain": _leer(ds, "TRAIN"), "val": _leer(ds, "VAL"), "test": _leer(ds, parte_test)}
    todo = pd.concat(partes.values(), ignore_index=True)
    ts = pd.to_datetime(todo["time:timestamp"])
    out = pd.DataFrame({
        "caseid": todo["Case ID"].astype(str),
        "task": todo["Activity"],
        "user": todo["Resource"].fillna("UNKNOWN").astype(str),
        "start_timestamp": ts,
        "end_timestamp": ts,
        "label": todo["label"],
    })
    out["_orden"] = range(len(out))   # se conserva el orden de eventos del archivo original
    out = out.sort_values(["caseid", "start_timestamp", "_orden"], kind="mergesort").drop(columns="_orden")
    destino = AQUI / log
    (destino / "split").mkdir(parents=True, exist_ok=True)
    out.to_csv(destino / "log.csv", index=False)
    ids = {k: sorted(v["Case ID"].astype(str).unique()) for k, v in partes.items()}
    solapados = (set(ids["subtrain"]) & set(ids["val"])) | (set(ids["subtrain"]) & set(ids["test"])) \
        | (set(ids["val"]) & set(ids["test"]))
    if solapados:
        raise ValueError(f"{log}: {len(solapados)} casos aparecen en mas de una division")
    ids["train"] = sorted(set(ids["subtrain"]) | set(ids["val"]))
    for nombre, lista in ids.items():
        (destino / "split" / f"{nombre}_caseids.txt").write_text("\n".join(lista) + "\n")
    largo = out.groupby("caseid").size()
    print(f"[ok] {log}: {len(ids['subtrain'])} subtrain | {len(ids['val'])} val | "
          f"{len(ids['test'])} test | {out['task'].nunique()} actividades | "
          f"largo medio {largo.mean():.1f}, maximo {largo.max()} -> {destino}")


if __name__ == "__main__":
    for log, (ds, parte_test) in LOGS.items():
        preparar(log, ds, parte_test)
