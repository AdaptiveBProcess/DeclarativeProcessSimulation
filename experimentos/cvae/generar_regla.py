"""
Genera logs con un CVAE entrenado con entrenar.py (logs de Dynamics, etiqueta de una
regla), usando su propio codigo (evaluation/generate_log.py). Para los modelos
publicados de CVAE (BPIC2012, Traffic Fines, Sepsis) se usa generar.py.

Dos modos:
  - CVAE (su metodo, por defecto): genera n trazas con la etiqueta 'deviant' (= la
    regla se cumple) y no verifica la regla. Mide el control que aprendio.
  - CVAE con rechazo (--rechazo): genera lotes de n trazas 'deviant' y guarda solo las
    que cumplen la regla, hasta tener n o agotar el presupuesto (10 veces n trazas
    generadas), igual que Dynamics (corregido) y RULE-GAN.
La regla se evalua con evaluacion.metricas_cambio.cumple_regla, en el orden en que
CVAE genera los eventos.

n = numero de casos de test comunes (107 en RunningExample).

Uso (entorno conda cvae):
    python experimentos/cvae/generar_regla.py --log RunningExample \
        --regla r1_directly_TaskB_TaskC --corridas 5 [--rechazo]

Cada corrida queda en experimentos/cvae/ejecuciones/<log>/<regla>[_rechazo]/corrida_XX/:
    <log>.csv            formato comun: caseid, task, user (vacio: no genera recursos),
                         start_timestamp = end_timestamp (no genera duraciones), label
    salida_cvae*.csv     salida original de CVAE (un archivo por lote con --rechazo)
    meta.json            incluye intentos (trazas generadas) y cuantas cumplian
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
REPO = AQUI / "repo_icpm2024"
COMUN = RAIZ / "experimentos" / "comun"
PRESUPUESTO = 10


def _modelo(carpeta_modelo: Path, info: dict):
    import torch
    from model import VAE
    from generar import HIPERPARAMETROS as h
    checkpoint = torch.load(carpeta_modelo / "modelo.pt", map_location="cpu")
    model = VAE(trace_attributes=info["TRACE_ATTRIBUTES"], num_activities=info["NUM_ACTIVITIES"],
                max_trace_length=info["MAX_TRACE_LENGTH"], num_lstm_layers=h["NUM_LSTM_LAYERS"],
                attr_e_dim=h["ATTR_E_DIM"], act_e_dim=h["ACT_E_DIM"], cf_dim=h["CF_DIM"],
                c_dim=info["NUM_LABELS"], z_dim=h["Z_DIM"], dropout_p=h["DROPOUT_P"]).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _lote(model, info, n: int, carpeta: Path, nombre: str):
    """Genera n trazas 'deviant'; devuelve el log en formato comun (orden de generacion)."""
    import pandas as pd
    from evaluation.generate_log import generate_log
    from generar import _formato_comun
    generate_log(model, generation_config={"deviant": n}, dataset_info=info,
                 output_path=str(carpeta), output_name=nombre)
    _formato_comun(carpeta / f"{nombre}.csv", carpeta / f"{nombre}_comun.csv")
    return pd.read_csv(carpeta / f"{nombre}_comun.csv", dtype={"caseid": str})


def generar(log: str, regla_id: str, corridas: int, semilla: int, rechazo: bool) -> None:
    import numpy as np
    import pandas as pd
    import torch
    sys.path.append(str(RAIZ))
    from evaluacion.metricas_cambio import cumple_regla
    from support_modules import traces_evaluation as te
    import entrenar as ent

    carpeta_modelo = AQUI / "modelos" / log / regla_id
    info = ent.dataset_info(AQUI / "datos" / log / regla_id)
    model = _modelo(carpeta_modelo, info)
    entrenamiento = json.loads((carpeta_modelo / "entrenamiento.json").read_text(encoding="utf-8"))
    ini = COMUN / log / "reglas" / f"{regla_id}.ini"
    regla = te.extract_rules(path=str(ini))
    n = len((COMUN / log / "split" / "test_caseids.txt").read_text().split())

    for i in range(1, corridas + 1):
        sem = semilla + i - 1
        random.seed(sem)
        np.random.seed(sem)
        torch.manual_seed(sem)
        destino = AQUI / "ejecuciones" / log / (regla_id + ("_rechazo" if rechazo else "")) / f"corrida_{i:02d}"
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir(parents=True)
        inicio = datetime.datetime.now()

        if not rechazo:
            df = _lote(model, info, n, destino, "salida_cvae")
            tr = df.groupby("caseid", sort=False)["task"].apply(list)
            conteo = {"intentos": n, "vacias": n - len(tr),
                      "intentos_cumplen": int(sum(cumple_regla(t, regla) for t in tr))}
        else:
            partes, aceptadas, intentos, cumplen, vacias, k = [], 0, 0, 0, 0, 0
            while aceptadas < n and intentos < PRESUPUESTO * n:
                lote = _lote(model, info, n, destino, f"salida_cvae_{k:02d}")
                tr = lote.groupby("caseid", sort=False)["task"].apply(list)
                pos = {c: int(c[len("GEN"):]) for c in tr.index}   # indice de generacion
                ultimo = n - 1
                for caso, tareas in tr.items():
                    if cumple_regla(tareas, regla):
                        cumplen += 1
                        aceptadas += 1
                        partes.append(lote[lote["caseid"] == caso].assign(caseid=f"GEN{k:02d}_{pos[caso]:04d}"))
                        if aceptadas == n:
                            ultimo = pos[caso]
                            break
                examinadas = ultimo + 1
                vacias += examinadas - sum(1 for c in tr.index if pos[c] <= ultimo)
                intentos += examinadas
                k += 1
            df = pd.concat(partes) if partes else pd.DataFrame(
                columns=["caseid", "task", "user", "start_timestamp", "end_timestamp", "label"])
            conteo = {"intentos": intentos, "vacias": vacias, "intentos_cumplen": cumplen,
                      "lotes": k}
            if aceptadas < n:
                print(f"[aviso] {regla_id}: {aceptadas} de {n} trazas dentro del presupuesto "
                      f"(corrida {i}).")

        segundos = round((datetime.datetime.now() - inicio).total_seconds(), 1)
        df.to_csv(destino / f"{log}.csv", index=False)
        shutil.copy(ini, destino / "regla.ini")
        meta = {
            "enfoque": "cvae" + ("_rechazo" if rechazo else ""), "log": log, "regla": regla_id,
            "corrida": i, "semilla": sem, "codigo": ent.COMMIT,
            "epoca_modelo": entrenamiento["epoca_elegida"],
            "segundos_entrenamiento": entrenamiento["segundos"],
            "presupuesto": PRESUPUESTO if rechazo else None,
            "inicio": inicio.isoformat(timespec="seconds"), "segundos": segundos,
            **conteo,
            "n_trazas": int(df["caseid"].nunique()), "n_eventos": int(len(df)),
        }
        (destino / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] {log} / {regla_id} / corrida {i}: {conteo} -> {destino}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--regla", required=True)
    ap.add_argument("--corridas", type=int, default=1)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--rechazo", action="store_true",
                    help="guardar solo las trazas que cumplen la regla (presupuesto 10x)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(AQUI))
    generar(args.log, args.regla, args.corridas, args.semilla, args.rechazo)
