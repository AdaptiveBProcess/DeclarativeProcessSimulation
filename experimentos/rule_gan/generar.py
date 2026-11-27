"""
Genera logs con un modelo RULE-GAN entrenado, bajo una regla.

Usa la configuracion elegida en la validacion (buscar_grilla.py): un checkpoint
(--epoca) y una temperatura de muestreo (--temperatura), con la misma decodificacion
que la seleccion (seleccion.generar_libre: corta en 'end', salta 'start', reescala
duracion y espera). La regla se impone por rechazo: se genera la traza completa y se
guarda solo si cumple la regla (o si todavia faltan trazas que no la cumplen, cuando
la regla pide una proporcion menor a 1).

Para que la comparacion con Dynamics sea pareja:
  - genera tantas trazas como casos de test comunes (107 en RunningExample);
  - la proporcion pedida sale de la 'variation' de la regla (=1 -> 100% cumple);
  - el presupuesto es 10 veces el numero de trazas pedidas (como Dynamics corregido);
  - la primera traza arranca en el inicio del test, como en Dynamics.

La regla se evalua con evaluacion.metricas_cambio.cumple_regla (evaluate_condition_list
sobre la secuencia de actividades), la misma semantica que usan las metricas.

Uso:
    python experimentos/rule_gan/generar.py --log RunningExample \
        --regla r1_directly_TaskB_TaskC --modelo 20261127_180201 --epoca 801 \
        --temperatura 0.5 --corridas 5

Cada corrida queda en experimentos/rule_gan/ejecuciones/<log>/<regla>/corrida_XX/:
    <log>.csv            caseid, task, user (= rol generado), start_timestamp, end_timestamp
    regla.ini, meta.json (incluye intentos, vacias y rechazadas)
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
COMUN = RAIZ / "experimentos" / "comun"
PRESUPUESTO = 10


def _modelo(log: str, nombre: str = None) -> Path:
    base = AQUI / "modelos" / log
    if nombre:
        return base / nombre
    carpetas = [p for p in base.iterdir() if (p / "parameters" / "model_parameters.json").exists()]
    if not carpetas:
        sys.exit(f"[error] No hay modelos entrenados en {base}; correr entrenar.py")
    return max(carpetas, key=lambda p: p.name)


def _inicio_test(log: str):
    import pandas as pd
    df = pd.read_csv(COMUN / log / "log.csv")
    test = set((COMUN / log / "split" / "test_caseids.txt").read_text().split())
    df = df[df["caseid"].astype(str).isin(test)]
    return pd.to_datetime(df["start_timestamp"], utc=True).dt.tz_convert(None).min(), len(test)


def aceptacion(regla: dict, n: int, p_pedida: float):
    """Funcion de aceptacion con metas por clase: round(p * n) trazas que cumplen y el
    resto que no cumple (con p = 1, solo se aceptan las que cumplen)."""
    from evaluacion.metricas_cambio import cumple_regla
    meta_si = int(round(p_pedida * n))
    cuenta = {"si": 0, "no": 0}

    def acepta(tareas):
        if cumple_regla(tareas, regla):
            if cuenta["si"] < meta_si:
                cuenta["si"] += 1
                return True
            return False
        if cuenta["no"] < n - meta_si:
            cuenta["no"] += 1
            return True
        return False

    return acepta


def generar(log: str, regla_id: str, corridas: int, semilla: int, modelo: str,
            epoca: int, temperatura: float) -> None:
    import seleccion as sel
    from support_modules import traces_evaluation as te

    carpeta = _modelo(log, modelo)
    epoca = epoca or max(sel.epocas_disponibles(carpeta))
    gen, mp = sel.cargar_generador(carpeta, epoca)
    ent_path = carpeta / "entrenamiento.json"
    entrenamiento = json.loads(ent_path.read_text(encoding="utf-8")) if ent_path.exists() else {}
    ini = COMUN / log / "reglas" / f"{regla_id}.ini"
    regla = te.extract_rules(path=str(ini))
    p_pedida = regla["prop_variation"] if regla.get("variation") == "=" else None
    if p_pedida is None:
        sys.exit("[error] La regla debe fijar la proporcion con 'variation = =p'")
    inicio_test, n_test = _inicio_test(log)

    for i in range(1, corridas + 1):
        sem = semilla + i - 1
        destino = AQUI / "ejecuciones" / log / regla_id / f"corrida_{i:02d}"
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir(parents=True)

        inicio = datetime.datetime.now()
        contadores = {}
        df, _ = sel.generar_libre(gen, mp, n_test, temperatura, sem, inicio_test,
                                  presupuesto=PRESUPUESTO,
                                  acepta=aceptacion(regla, n_test, p_pedida),
                                  contadores=contadores)
        segundos = round((datetime.datetime.now() - inicio).total_seconds(), 1)
        if df.empty:
            print(f"[aviso] {regla_id}: ninguna traza generada cumple la regla dentro del "
                  f"presupuesto de intentos (corrida {i}).")
        elif contadores["aceptadas"] < n_test:
            print(f"[aviso] {regla_id}: {contadores['aceptadas']} de {n_test} trazas dentro "
                  f"del presupuesto (corrida {i}).")
        df.to_csv(destino / f"{log}.csv", index=False)
        shutil.copy(ini, destino / "regla.ini")
        meta = {
            "enfoque": "rule_gan", "log": log, "regla": regla_id, "corrida": i,
            "semilla": sem, "modelo": carpeta.name, "epoca": epoca,
            "temperatura": temperatura, "conjunto_entrenamiento": entrenamiento.get("conjunto"),
            "segundos_entrenamiento": entrenamiento.get("segundos"),
            "p_pedida": p_pedida, "presupuesto": PRESUPUESTO,
            "inicio": inicio.isoformat(timespec="seconds"), "segundos": segundos,
            **contadores,
            "n_trazas": int(df["caseid"].nunique()) if len(df) else 0,
            "n_eventos": int(len(df)),
        }
        (destino / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] {log} / {regla_id} / corrida {i}: {contadores} -> {destino}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--regla", required=True)
    ap.add_argument("--corridas", type=int, default=1)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--modelo", default=None, help="carpeta del modelo (por defecto, la mas reciente)")
    ap.add_argument("--epoca", type=int, default=None, help="checkpoint (por defecto, el ultimo)")
    ap.add_argument("--temperatura", type=float, required=True,
                    help="la elegida en la validacion (busqueda/<log>/grilla/ensayos.csv)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.chdir(RAIZ)
    sys.path.insert(0, str(RAIZ))
    sys.path.insert(0, str(AQUI))
    generar(args.log, args.regla, args.corridas, args.semilla, args.modelo, args.epoca,
            args.temperatura)
