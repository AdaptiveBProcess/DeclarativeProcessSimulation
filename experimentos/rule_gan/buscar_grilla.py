"""
Busqueda en GRILLA de la configuracion de RULE-GAN para un log, con la validacion
(Fase 1b de la metodologia de seleccion). Se elige una configuracion por log. El test
no se usa.

Espacio: 12 configuraciones, se prueban todas. Se redujo a partir de la busqueda amplia
en RunningExample y PurchasingExample (buscar_amplia.py, 30 configuraciones por log;
comentarios-cloud/CLOUD.md, seccion 29.7).

  Variables
    num_heads   {2, 4}         la mejor depende del log (RE: 4, PE: 2)
    ff_dim      {128, 256}     64 fue la peor en ambos logs
    tasas de Adam (lr_d del critico, lr_g del generador):
        (1e-4, 2e-4)   las de V2-V6
        (1e-4, 1e-4)   iguales, el valor por defecto de WGAN-GP (Gulrajani et al. 2017)
        (4e-4, 1e-4)   critico mas rapido, como en SAGAN
  Fijos (ver FIJOS y entrenar.HIPERPARAMETROS)
    d_model 64, num_blocks 3, time2vec_dim 16        mejores en ambos logs
    bg_lambda 0, ct_lambda 0                         sin perdidas auxiliares (0 fue lo mejor
                                                     en ambos logs; ct_lambda 1, lo peor)
    betas (0, 0.9), dropout 0.1, latent_dim 64, batch 32, n_critic 5, gp_lambda 10,
    lognorm, rp_sim 0.85, 1000 epocas con checkpoints cada 100

Cada configuracion se evalua como en la busqueda amplia (seleccion.evaluar_modelo): la
epoca con T = 1 y luego la temperatura {0.5, 0.6, 0.7, 0.85, 1.15, 1.3} en esa epoca, con
5 logs generados del tamano de la validacion. Objetivo: promedio de las metricas
relativas a V4 (menor es mejor). V4 va primero porque fija la referencia.

Reutilizacion: si una configuracion ya se evaluo en una busqueda anterior del mismo log
(--reutilizar; por defecto la busqueda amplia 'seleccion', donde la tasa era la de
V2-V6), se copian sus evaluaciones y solo se evaluan las temperaturas que falten, en su
mejor epoca con T = 1. Se usa la misma referencia de V4 y el mismo modelo DECLARE de esa
busqueda. Requiere su modelo en busqueda/<log>/<reutilizar>/modelos/.

Resultado en experimentos/rule_gan/busqueda/<log>/<etiqueta>/ (por defecto 'grilla'):
    ensayos.csv        una fila por configuracion (ensayo = posicion en la grilla)
    evaluaciones.csv   todas las evaluaciones (epoca x temperatura)
    referencia_v4.json, modelo_conf.pkl, modelos/ (no versionados los dos ultimos)

Uso:
    python experimentos/rule_gan/buscar_grilla.py --log RunningExample --plan   # solo el plan
    python experimentos/rule_gan/buscar_grilla.py --log RunningExample
"""
import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import entrenar as ent  # noqa: E402

VARIABLES = {
    "num_heads": [2, 4],
    "ff_dim": [128, 256],
    "tasas": [(1e-4, 2e-4), (1e-4, 1e-4), (4e-4, 1e-4)],   # (lr_d, lr_g)
}
FIJOS = {"d_model": 64, "num_blocks": 3, "time2vec_dim": 16, "bg_lambda": 0.0, "ct_lambda": 0.0}
TASAS_V2_V6 = (1e-4, 2e-4)
CLAVES = ["d_model", "num_heads", "num_blocks", "ff_dim", "time2vec_dim",
          "bg_lambda", "ct_lambda", "lr_d", "lr_g"]
METRICAS = ["RED", "CTD", "2GD", "freq_act", "1-CONF", "div_gap"]
RESULTADOS = (METRICAS + [m + "_std" for m in METRICAS]
              + ["pct_vacias", "pct_vacias_std", "objetivo"])
COLS_EVAL = ["ensayo", "version", "modelo", *CLAVES, "epoca", "temperatura", *RESULTADOS]
COLS_ENSAYO = ["ensayo", "version", "modelo", *CLAVES, "mejor_epoca", "mejor_temperatura",
               *RESULTADOS, "seg_entrenamiento", "seg_total", "reutilizado_de"]


def grilla() -> list:
    """Las 12 configuraciones, con V4 primero."""
    v4 = {**FIJOS, "num_heads": 4, "ff_dim": 256, "lr_d": TASAS_V2_V6[0], "lr_g": TASAS_V2_V6[1]}
    resto = [{**FIJOS, "num_heads": h, "ff_dim": f, "lr_d": d, "lr_g": g}
             for h, f, (d, g) in itertools.product(*VARIABLES.values())]
    resto.remove(v4)
    return [v4] + resto


def clave(c) -> str:
    return json.dumps({k: (float(c[k]) if k in ("bg_lambda", "ct_lambda", "lr_d", "lr_g")
                           else int(c[k])) for k in CLAVES}, sort_keys=True)


def previas(carpeta: Path) -> dict:
    """clave -> fila de ensayos.csv de una busqueda. En la busqueda amplia no hay
    columnas de tasa: era la de V2-V6."""
    import pandas as pd
    if carpeta is None or not (carpeta / "ensayos.csv").exists():
        return {}
    d = pd.read_csv(carpeta / "ensayos.csv")
    if "lr_d" not in d:
        d["lr_d"], d["lr_g"] = TASAS_V2_V6
    return {clave(f): f for _, f in d.iterrows()}


def evaluaciones_previas(origen: Path, modelo: str) -> list:
    import pandas as pd
    d = pd.read_csv(origen / "evaluaciones.csv")
    d = d[d["modelo"] == modelo]
    return [{k: r[k] for k in ["epoca", "temperatura", *RESULTADOS]} for r in d.to_dict("records")]


def temperaturas_faltantes(filas: list, temperaturas) -> tuple:
    """Mejor epoca con T = 1 y las temperaturas que aun no se evaluaron en ella."""
    con_t1 = [f for f in filas if float(f["temperatura"]) == 1.0]
    epoca = int(min(con_t1, key=lambda f: f["objetivo"])["epoca"])
    ya = {float(f["temperatura"]) for f in filas if int(f["epoca"]) == epoca}
    return epoca, [t for t in temperaturas if t not in ya]


def _anexar(archivo: Path, filas: list, columnas: list) -> None:
    import pandas as pd
    pd.DataFrame(filas)[columnas].to_csv(archivo, mode="a", index=False,
                                         header=not archivo.exists())


def buscar(log: str, etiqueta: str, reutilizar: str, semilla: int, n_gen: int,
           epocas: int, cada: int, solo_plan: bool) -> None:
    import pandas as pd
    import seleccion as sel

    salida = AQUI / "busqueda" / log / etiqueta
    origen = AQUI / "busqueda" / log / reutilizar if reutilizar else None
    hechas, reusables = previas(salida), previas(origen)

    plan = []
    for i, c in enumerate(grilla(), 1):
        nombre = "V4" if i == 1 else f"G{i:02d}"
        fila = reusables.get(clave(c))
        if clave(c) in hechas:
            accion, detalle = "hecha", ""
        elif fila is not None and (origen / "modelos" / fila["modelo"]).exists():
            epoca, faltan = temperaturas_faltantes(evaluaciones_previas(origen, fila["modelo"]),
                                                   sel.TEMPERATURAS)
            accion, detalle = "reutiliza", f"{reutilizar}/{fila['modelo']}, epoca {epoca}, T {faltan}"
        else:
            accion, detalle = "entrenar", ""
        plan.append((i, nombre, c, accion, fila))
        print(f"  {nombre:>3}  heads {c['num_heads']}  ff {c['ff_dim']:>3}  "
              f"lr_d {c['lr_d']:g}  lr_g {c['lr_g']:g}  -> {accion} {detalle}")
    n = {a: sum(p[3] == a for p in plan) for a in ("entrenar", "reutiliza", "hecha")}
    print(f"[plan] {log}: {n['entrenar']} a entrenar, {n['reutiliza']} reutilizadas, "
          f"{n['hecha']} ya hechas")
    if solo_plan:
        return

    salida.mkdir(parents=True, exist_ok=True)
    for archivo in ("referencia_v4.json", "modelo_conf.pkl"):
        if origen is not None and (origen / archivo).exists() and not (salida / archivo).exists():
            shutil.copy(origen / archivo, salida / archivo)
    ref = sel.referencia(log, salida / "modelo_conf.pkl")
    archivo_ref = salida / "referencia_v4.json"
    ref_v4 = json.loads(archivo_ref.read_text()) if archivo_ref.exists() else None

    for i, nombre, c, accion, fila in plan:
        if accion == "hecha":
            continue
        if ref_v4 is None and nombre != "V4":
            raise RuntimeError("Falta la referencia de V4: V4 debe evaluarse primero.")
        print(f"\n[grilla] {log} | {nombre} ({i}/12, {accion}) | {c}", flush=True)
        t0 = time.time()
        if accion == "entrenar":
            carpeta = ent.entrenar(log, {**c, "epochs": epocas}, semilla=semilla,
                                   conjunto="subtrain", cada=cada, salida=salida / "modelos")
            t_entr = time.time() - t0
            filas, mejor, nueva_ref = sel.evaluar_modelo(carpeta, ref, ref_v4, n_gen,
                                                         sel.TEMPERATURAS, semilla)
            if ref_v4 is None:
                ref_v4 = nueva_ref
                archivo_ref.write_text(json.dumps(ref_v4, indent=2))
            modelo, reutilizado, t_previo = carpeta.name, "", 0.0
        else:
            carpeta = origen / "modelos" / fila["modelo"]
            filas = evaluaciones_previas(origen, fila["modelo"])
            for f in filas:
                f["objetivo"] = sel.objetivo(f, ref_v4)
            epoca, faltan = temperaturas_faltantes(filas, sel.TEMPERATURAS)
            filas += sel.evaluar_temperaturas(carpeta, epoca, ref, ref_v4, faltan, n_gen, semilla)
            mejor = min(filas, key=lambda f: f["objetivo"])
            modelo, reutilizado = fila["modelo"], reutilizar
            t_entr, t_previo = float(fila["seg_entrenamiento"]), float(fila["seg_total"])
        base = {"ensayo": i, "version": nombre, "modelo": modelo, **c}
        _anexar(salida / "evaluaciones.csv", [{**base, **f} for f in filas], COLS_EVAL)
        _anexar(salida / "ensayos.csv", [{
            **base, "mejor_epoca": mejor["epoca"], "mejor_temperatura": mejor["temperatura"],
            **{k: mejor[k] for k in RESULTADOS},
            "seg_entrenamiento": round(t_entr, 1),
            "seg_total": round(t_previo + time.time() - t0, 1),
            "reutilizado_de": reutilizado}], COLS_ENSAYO)
        print(f"[grilla] objetivo={mejor['objetivo']:.4f} (epoca {mejor['epoca']}, "
              f"T={mejor['temperatura']}) en {time.time() - t0:.0f}s", flush=True)

    res = pd.read_csv(salida / "ensayos.csv").sort_values("objetivo")
    print(f"\n[ok] Grilla {log} terminada ({salida / 'ensayos.csv'}):")
    print(res[["version", "num_heads", "ff_dim", "lr_d", "lr_g", "mejor_epoca",
               "mejor_temperatura", "objetivo"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--etiqueta", default="grilla")
    ap.add_argument("--reutilizar", default="seleccion",
                    help="busqueda anterior del mismo log cuyos resultados se reutilizan ('' = ninguna)")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--n-gen", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=1000)
    ap.add_argument("--cada", type=int, default=100)
    ap.add_argument("--plan", action="store_true", help="solo muestra que se entrena y que se reutiliza")
    args = ap.parse_args()
    ent.preparar_entorno()
    buscar(args.log, args.etiqueta, args.reutilizar, args.semilla, args.n_gen, args.epocas,
           args.cada, args.plan)
