# -*- coding: utf-8 -*-
"""
Estudio de varianza ENTRE entrenamientos independientes de GANTrainerV2
(transformer_wgan) sobre RunningExample.

Por que este estudio
---------------------
Hasta ahora solo medimos varianza de MUESTREO de generacion: 10 logs generados
por el MISMO modelo, entrenado una sola vez. Esto NO dice nada sobre que tan
distinto sale un modelo si se entrena de nuevo con otra semilla (WGAN-GP es
conocido por su inestabilidad run-to-run). Este script corre N entrenamientos
completos e independientes, cada uno evaluado con el pipeline normal (10
replicas de generacion via dg_prediction.py), y compara la media de cada
metrica ENTRE esas N corridas de entrenamiento.

No modifica dg_training.py ni dg_prediction.py -- los invoca como subprocesos
separados (mismo patron que correrlos manualmente), para que cada entrenamiento
arranque en un proceso limpio sin arrastrar estado de TensorFlow entre corridas.

Uso (desde la raiz del repo, en el venv deep_generator):
    python estudio_varianza_transformer_wgan.py        # N=10 por defecto
    python estudio_varianza_transformer_wgan.py 15      # N=15
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

METRICAS_DIR = Path("data/4.simulation_results/RunningExample/metricas")
METRIC_COLS = ["RED", "CTD", "CTD_horas", "2GD", "CONF"]


def _run(cmd, label):
    print(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}")
    subprocess.run(cmd, check=True)
    print(f"[OK] {label} completado")


def _find_new_summary(before: set) -> Path:
    after = set(METRICAS_DIR.glob("metrics_summary_*.csv"))
    new = after - before
    if not new:
        raise RuntimeError(
            "No aparecio ningun metrics_summary_*.csv nuevo tras "
            "dg_prediction.py -- revisa la salida de consola arriba.")
    # Si por alguna razon aparece mas de uno, tomar el mas reciente
    return max(new, key=lambda p: p.stat().st_mtime)


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    py = sys.executable

    print(f"Estudio de varianza: {n_runs} entrenamientos independientes "
          f"de transformer_wgan sobre RunningExample.\n")

    filas = []
    fallos = []

    for i in range(1, n_runs + 1):
        print(f"\n\n########## CORRIDA DE ENTRENAMIENTO {i}/{n_runs} ##########")
        t0 = time.time()
        try:
            before = set(METRICAS_DIR.glob("metrics_summary_*.csv"))
            _run([py, "dg_training.py", "-m", "transformer_wgan"],
                 f"Entrenamiento {i}/{n_runs}")
            _run([py, "dg_prediction.py"], f"Prediccion {i}/{n_runs}")

            summary_path = _find_new_summary(before)
            df_run = pd.read_csv(summary_path, index_col=0)
            fila = {c: df_run.loc["mean", c] for c in METRIC_COLS if c in df_run.columns}
            fila["corrida"] = i
            fila["elapsed_s"] = round(time.time() - t0, 1)
            fila["summary_file"] = summary_path.name
            filas.append(fila)
            print(f"[OK] Corrida {i} completa en {fila['elapsed_s']:.0f}s "
                  f"-- {summary_path.name}")

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            print(f"[FAIL] Corrida {i} fallo tras {elapsed:.0f}s: {exc}")
            fallos.append({"corrida": i, "error": str(exc), "elapsed_s": elapsed})
            continue

    if not filas:
        print("\n[ERROR] Ninguna corrida se completo exitosamente.")
        sys.exit(1)

    df_all = pd.DataFrame(filas).set_index("corrida")
    out_path = Path(
        f"estudio_varianza_transformer_wgan_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    df_all.to_csv(out_path)

    print("\n\n" + "=" * 70)
    print(f"RESUMEN DEL ESTUDIO DE VARIANZA ENTRE ENTRENAMIENTOS "
          f"({len(filas)}/{n_runs} corridas exitosas)")
    print("=" * 70)
    cols_mostrar = [c for c in METRIC_COLS if c in df_all.columns]
    print(df_all[cols_mostrar].to_string())

    print("\nEstadisticos ENTRE corridas de entrenamiento independientes "
          "(esto es lo nuevo -- antes solo teniamos varianza de muestreo "
          "de generacion dentro de UN solo modelo entrenado):")
    print(df_all[cols_mostrar].agg(["mean", "std", "min", "max"]).to_string())

    if fallos:
        print(f"\n[AVISO] {len(fallos)} corridas fallaron:")
        for f in fallos:
            print(f"  corrida {f['corrida']}: {f['error']}")

    print(f"\nGuardado en: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
