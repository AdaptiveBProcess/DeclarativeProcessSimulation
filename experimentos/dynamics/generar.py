"""
Genera logs con el LSTM publicado de Dynamics bajo una regla (sin reentrenar).

Replica los parametros del dg_prediction.py original de Dynamics (paquete de
Zenodo): activity='pred_log', variante 'Rules Based Random Choice', dos timestamps
por evento. Genera tantas trazas como casos tiene el test publicado del modelo.

Por defecto aplica la correccion B de la regla de aceptacion (ver
predictor_corregido.py): el codigo publicado genera ~1 traza por nucleo de CPU.
Con --publicado se corre tal cual esta publicado.

Como en el pipeline de Dynamics (support_modules/predictor_adapter.hallucinate),
se eliminan del log final los eventos 'start'/'end' que el LSTM puede muestrear.

Requisitos:
  - entorno conda deep_generator
  - submodulo GenerativeLSTM inicializado
  - modelos descargados: python experimentos/dynamics/descargar_paquete.py

Uso:
    python experimentos/dynamics/generar.py --log RunningExample \
        --regla r1_directly_TaskB_TaskC --corridas 3 --semilla 0

Cada corrida queda en experimentos/dynamics/ejecuciones/<log>/<regla>/corrida_XX/
(o .../<regla>_publicado/... con --publicado):
    <log>.csv              log generado (columna 'user' = rol generado por el LSTM)
    prediccion.csv         salida cruda del predictor
    similitud.csv          metricas internas del predictor de Dynamics
    regla.ini, meta.json   regla usada y parametros de la corrida
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
REGLAS = RAIZ / "experimentos" / "comun"


def _modelo(log: str) -> Path:
    carpetas = [p for p in (AQUI / "modelos" / log).iterdir() if p.is_dir()]
    if len(carpetas) != 1:
        sys.exit(f"[error] Se esperaba un modelo en {AQUI / 'modelos' / log}; "
                 f"correr descargar_paquete.py")
    return carpetas[0]


def _parametros(log: str, id_modelo: str, variante: str) -> dict:
    return {
        "filename": f"{log}.xes",
        "log_name": log,
        "activity": "pred_log",
        "folder": id_modelo,
        "model_file": f"{log}.h5",
        "is_single_exec": False,
        "variant": variante,
        "rep": 1,
        "one_timestamp": False,
        "include_org_log": False,
        "read_options": {
            "timeformat": "%Y-%m-%d %H:%M:%S%Z",
            "column_names": {"Case ID": "caseid", "Activity": "task",
                             "lifecycle:transition": "event_type", "Resource": "user"},
            "one_timestamp": False,
            "filter_d_attrib": False,
        },
    }


def _limpiar_modelo(carpeta: Path, log: str) -> dict:
    """El predictor escribe sus salidas dentro de la carpeta del modelo; se sacan
    de ahi para que el modelo publicado quede intacto entre corridas."""
    salidas = {}
    for p in carpeta.glob(f"gen_{log}_*.csv"):
        salidas["prediccion.csv"] = p
    for p in carpeta.glob("SE_*.csv"):
        salidas["similitud.csv"] = p
    for p in carpeta.glob(f"tst_{log}.csv"):
        p.unlink()
    trazas = carpeta / "parameters" / "traces_generated"
    if trazas.exists():
        shutil.rmtree(trazas)
    return salidas


def _intentos(carpeta: Path) -> dict:
    """Trazas generadas, cuantas cumplian la regla y cuantas se aceptaron, segun lo que
    anota predictor_corregido.py (una linea por traza generada). Borra las anotaciones."""
    carpeta_i = carpeta / "parameters" / "intentos"
    filas = []
    if carpeta_i.exists():
        for p in carpeta_i.glob("*.txt"):
            filas += [tuple(map(int, l.split(","))) for l in p.read_text().split()]
        shutil.rmtree(carpeta_i)
    return {"intentos": len(filas), "intentos_cumplen": sum(c for c, _ in filas),
            "intentos_aceptados": sum(a for _, a in filas)}


def _limpiar_log(destino: Path, log: str) -> dict:
    """Quita los eventos 'start'/'end' (como hallucinate en el pipeline de Dynamics)
    y borra el .xes, que el predictor escribe antes de ese filtrado."""
    import pandas as pd
    csv = destino / f"{log}.csv"
    df = pd.read_csv(csv)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[~df["task"].isin(["Start", "End", "start", "end"])]
    df.to_csv(csv, index=False)
    xes = destino / f"{log}.xes"
    if xes.exists():
        xes.unlink()
    return {"n_trazas": int(df["caseid"].nunique()), "n_eventos": int(len(df))}


def generar(log: str, regla: str, corridas: int, semilla: int, variante: str,
            publicado: bool) -> None:
    import numpy as np
    import tensorflow as tf
    from GenerativeLSTM.model_prediction import event_log_predictor as elp
    from GenerativeLSTM.model_prediction import model_predictor as pr

    if not publicado:
        from experimentos.dynamics.predictor_corregido import EventLogPredictorCorregido
        elp.EventLogPredictor = EventLogPredictorCorregido   # interfaces lo instancia via elp

    carpeta_modelo = _modelo(log)
    ini = REGLAS / log / "reglas" / f"{regla}.ini"
    if not ini.exists():
        sys.exit(f"[error] No existe la regla {ini}")

    for i in range(1, corridas + 1):
        sem = semilla + i - 1
        random.seed(sem)
        np.random.seed(sem)
        tf.random.set_seed(sem)

        carpeta_regla = regla + ("_publicado" if publicado else "")
        destino = AQUI / "ejecuciones" / log / carpeta_regla / f"corrida_{i:02d}"
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir(parents=True)

        _intentos(carpeta_modelo)              # descarta anotaciones de una corrida cortada
        inicio = datetime.datetime.now()
        pr.ModelPredictor(
            _parametros(log, carpeta_modelo.name, variante),
            input_folder=str(carpeta_modelo.parent),
            output_folder=str(destino),
            rules_path=str(ini),
        )
        segundos = round((datetime.datetime.now() - inicio).total_seconds(), 1)
        intentos = _intentos(carpeta_modelo)
        for nombre, p in _limpiar_modelo(carpeta_modelo, log).items():
            shutil.move(str(p), str(destino / nombre))
        shutil.copy(ini, destino / "regla.ini")
        meta = {
            "enfoque": "dynamics", "log": log, "regla": regla, "corrida": i,
            "semilla": sem, "variante": variante, "modelo": carpeta_modelo.name,
            "correccion_B": not publicado,
            "inicio": inicio.isoformat(timespec="seconds"),
            "segundos": segundos,
            **(intentos if not publicado else {}),
            **_limpiar_log(destino, log),
        }
        (destino / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] {log} / {regla} / corrida {i}: {destino}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--regla", required=True, help="nombre del .ini en experimentos/comun/<log>/reglas/")
    ap.add_argument("--corridas", type=int, default=1)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--variante", default="Rules Based Random Choice")
    ap.add_argument("--publicado", action="store_true",
                    help="correr el generador tal como esta publicado (sin la correccion B)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.chdir(RAIZ)                  # el predictor usa rutas relativas a la raiz del repo
    sys.path.insert(0, str(RAIZ))
    generar(args.log, args.regla, args.corridas, args.semilla, args.variante, args.publicado)
