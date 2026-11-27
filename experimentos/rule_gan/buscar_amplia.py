"""
Busqueda AMPLIA de la configuracion de RULE-GAN para un log, con la validacion
(Fase 1a de la metodologia de seleccion). El test no se usa.

Se corrio en RunningExample y PurchasingExample (busqueda/<log>/seleccion/). Con sus
resultados se redujo el espacio para todos los logs: ver buscar_grilla.py (Fase 1b) y
comentarios-cloud/CLOUD.md, seccion 29.7. Este script queda para reproducir esa fase;
la tasa de aprendizaje y las betas quedan en los valores de V2-V6.

Procedimiento:
  1. Primero se evaluan las versiones historicas, en este orden: V4 (fija la
     referencia del objetivo), V2, V3, V5, V6.
  2. Luego hyperopt (TPE, como el optimizador de Dynamics) hasta completar el
     presupuesto, sobre el espacio que abarca V2-V6.
  3. Cada ensayo: entrenar 1000 epocas con checkpoints cada 100, elegir la epoca con
     temperatura 1 y luego la temperatura, promediando 5 logs generados del tamano de
     la validacion (ver seleccion.py).
  4. El presupuesto cuenta configuraciones distintas. Si TPE propone una ya
     entrenada, se reutiliza su objetivo sin volver a entrenar.

Espacio (las tasas de aprendizaje, latent_dim=64, batch=32, n_critic=5 y
dropout=0.1 quedan fijos, como en V2-V6):
    d_model {32, 64} · num_heads {2, 4} · num_blocks {2, 3} · ff_dim {64, 128, 256}
    time2vec_dim {8, 16} · bg_lambda {0, 0.05, 5} · ct_lambda {0, 0.1, 1}

Resultado en experimentos/rule_gan/busqueda/<log>/<etiqueta>/:
    ensayos.csv        un ensayo por fila: configuracion, mejor epoca y temperatura,
                       objetivo y metricas de validacion
    evaluaciones.csv   todas las evaluaciones (epoca x temperatura) de cada ensayo
    referencia_v4.json metricas de V4 usadas para normalizar el objetivo
    trials.pkl         estado de hyperopt (permite reanudar)
    modelos/           modelos entrenados (no versionado)

Uso:
    python experimentos/rule_gan/buscar_amplia.py --log RunningExample --presupuesto 30
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import entrenar as ent  # noqa: E402

ESPACIO = {
    "d_model": [32, 64],
    "num_heads": [2, 4],
    "num_blocks": [2, 3],
    "ff_dim": [64, 128, 256],
    "time2vec_dim": [8, 16],
    "bg_lambda": [0.0, 0.05, 5.0],
    "ct_lambda": [0.0, 0.1, 1.0],
}
# Versiones historicas (comentarios-cloud/CLOUD.md, seccion 5)
V4 = {"d_model": 64, "num_heads": 4, "num_blocks": 3, "ff_dim": 256, "time2vec_dim": 16,
      "bg_lambda": 0.0, "ct_lambda": 0.0}
HISTORICAS = {
    "V4": V4,
    "V2": {"d_model": 32, "num_heads": 2, "num_blocks": 2, "ff_dim": 64, "time2vec_dim": 8,
           "bg_lambda": 0.0, "ct_lambda": 0.0},
    "V3": {"d_model": 32, "num_heads": 2, "num_blocks": 3, "ff_dim": 128, "time2vec_dim": 16,
           "bg_lambda": 0.0, "ct_lambda": 0.0},
    "V5": {**V4, "bg_lambda": 5.0, "ct_lambda": 1.0},
    "V6": {**V4, "bg_lambda": 0.05, "ct_lambda": 0.1},
}


def buscar(log: str, presupuesto: int, etiqueta: str, semilla: int, n_gen: int,
           temperaturas, epocas: int, cada: int) -> None:
    import numpy as np
    import pandas as pd
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    from hyperopt.fmin import generate_trials_to_calculate
    import seleccion as sel

    salida = AQUI / "busqueda" / log / etiqueta
    salida.mkdir(parents=True, exist_ok=True)
    ref = sel.referencia(log, salida / "modelo_conf.pkl")
    archivo_ref = salida / "referencia_v4.json"
    estado = {"ref_v4": json.loads(archivo_ref.read_text()) if archivo_ref.exists() else None}

    nombre_de = {json.dumps(c, sort_keys=True): n for n, c in HISTORICAS.items()}

    def _clave(config):
        config = {k: (float(config[k]) if k.endswith("lambda") else int(config[k])) for k in ESPACIO}
        return json.dumps(config, sort_keys=True)

    def evaluadas():
        """Objetivo de cada configuracion ya entrenada (clave -> objetivo)."""
        if not (salida / "ensayos.csv").exists():
            return {}
        previos = pd.read_csv(salida / "ensayos.csv")
        return {_clave(f): float(f["objetivo"]) for _, f in previos.iterrows()}

    def objetivo(config):
        config = {k: (float(v) if k.endswith("lambda") else int(v)) for k, v in config.items()}
        clave = json.dumps(config, sort_keys=True)
        nombre = nombre_de.get(clave, "TPE")
        previas = evaluadas()
        if clave in previas:
            # Misma configuracion y misma semilla dan el mismo resultado: no se
            # vuelve a entrenar ni cuenta para el presupuesto.
            print(f"[busqueda] {log} | repetida ({nombre}), se reutiliza: {config}", flush=True)
            return {"loss": previas[clave], "status": STATUS_OK, "repetida": True}
        n_ensayo = len(pd.read_csv(salida / "ensayos.csv")) + 1 if (salida / "ensayos.csv").exists() else 1
        print(f"\n[busqueda] {log} | ensayo {n_ensayo}/{presupuesto} ({nombre}) | {config}", flush=True)
        t0 = time.time()
        carpeta = ent.entrenar(log, {**config, "epochs": epocas}, semilla=semilla,
                               conjunto="subtrain", cada=cada, salida=salida / "modelos")
        t_entr = time.time() - t0
        filas, mejor, ref_v4 = sel.evaluar_modelo(carpeta, ref, estado["ref_v4"], n_gen,
                                                  temperaturas, semilla)
        if estado["ref_v4"] is None:
            estado["ref_v4"] = ref_v4
            archivo_ref.write_text(json.dumps(ref_v4, indent=2))
        base = {"ensayo": n_ensayo, "version": nombre, "modelo": carpeta.name, **config}
        pd.DataFrame([{**base, **f} for f in filas]).to_csv(
            salida / "evaluaciones.csv", mode="a", index=False,
            header=not (salida / "evaluaciones.csv").exists())
        fila = {**base, **{f"mejor_{k}" if k in ("epoca", "temperatura") else k: v
                           for k, v in mejor.items()},
                "seg_entrenamiento": round(t_entr, 1),
                "seg_total": round(time.time() - t0, 1)}
        pd.DataFrame([fila]).to_csv(salida / "ensayos.csv", mode="a", index=False,
                                    header=not (salida / "ensayos.csv").exists())
        print(f"[busqueda] objetivo={mejor['objetivo']:.4f} (epoca {mejor['epoca']}, "
              f"T={mejor['temperatura']}) en {fila['seg_total']:.0f}s", flush=True)
        return {"loss": mejor["objetivo"], "status": STATUS_OK}

    espacio = {k: hp.choice(k, v) for k, v in ESPACIO.items()}
    archivo_trials = salida / "trials.pkl"
    if archivo_trials.exists():
        trials = pickle.loads(archivo_trials.read_bytes())
    else:
        # hyperopt espera el indice de cada opcion en los puntos iniciales
        puntos = [{k: ESPACIO[k].index(c[k]) for k in ESPACIO} for c in HISTORICAS.values()]
        trials = generate_trials_to_calculate(puntos)
    rng = np.random.default_rng(semilla)

    def completados():
        return sum(t["result"].get("status") == STATUS_OK for t in trials.trials)

    # Un ensayo por llamada, guardando el estado despues de cada una. La primera
    # llamada evalua las versiones historicas pendientes. El presupuesto cuenta
    # configuraciones distintas; las propuestas de TPE se limitan a 4x el
    # presupuesto por si TPE solo propone configuraciones ya evaluadas.
    while len(evaluadas()) < presupuesto and len(trials.trials) < 4 * presupuesto:
        fmin(objetivo, espacio, algo=tpe.suggest, max_evals=completados() + 1, trials=trials,
             rstate=rng, show_progressbar=False)
        archivo_trials.write_bytes(pickle.dumps(trials))
    print(f"\n[ok] Busqueda {log} terminada: {len(evaluadas())} configuraciones distintas en "
          f"{len(trials.trials)} propuestas: {salida / 'ensayos.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--presupuesto", type=int, default=30)
    ap.add_argument("--etiqueta", default="seleccion")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--n-gen", type=int, default=5)
    ap.add_argument("--temperaturas", type=float, nargs="+", default=[0.7, 0.85, 1.15, 1.3])
    ap.add_argument("--epocas", type=int, default=1000)
    ap.add_argument("--cada", type=int, default=100)
    args = ap.parse_args()
    ent.preparar_entorno()
    buscar(args.log, args.presupuesto, args.etiqueta, args.semilla, args.n_gen,
           args.temperaturas, args.epocas, args.cada)
