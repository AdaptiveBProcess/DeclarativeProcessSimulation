"""
Entrena RULE-GAN (GANTrainerV2: Transformer + Time2Vec + WGAN-GP) con la division
comun y guarda checkpoints para elegir la epoca con la validacion.

Diferencias con dg_training.py -m transformer_wgan (el codigo del entrenador no
se modifica):
  - Division: se entrena con experimentos/comun/<log>/split/subtrain_caseids.txt
    (train sin la validacion). El test y la validacion no se usan para entrenar.
  - Hiperparametros: por defecto los de V4; se pueden cambiar (ver HIPERPARAMETROS)
    para las busquedas de experimentos/rule_gan/buscar_amplia.py y buscar_grilla.py.
  - Checkpoints: se guardan los pesos del generador cada `cada` epocas en
    <modelo>/checkpoints/epoca_XXXX.h5 (XXXX = epocas completadas), ademas del
    generador final que guarda el entrenador.

Las tasas de aprendizaje y las betas de Adam son configurables (lr_d, lr_g, beta_1,
beta_2). Por defecto valen lo mismo que en V2-V6: critico 1e-4, generador 2e-4,
betas (0, 0.9).

El entrenamiento no depende de la regla. En el hold-out, --excluir-regla quita del
entrenamiento los casos que cumplen la regla.

Uso:
    python experimentos/rule_gan/entrenar.py --log RunningExample
    python experimentos/rule_gan/entrenar.py --log RunningExample --config '{"d_model": 32}'

Resultado: experimentos/rule_gan/modelos/<log>/<AAAAMMDD_HHMMSS>/
    gen/ (generador final), checkpoints/, parameters/, entrenamiento.json
"""
import argparse
import builtins
import json
import os
import random
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
COMUN = RAIZ / "experimentos" / "comun"

# V4: configuracion "v4 CPU" de dg_training.py (transformer_wgan)
HIPERPARAMETROS = {
    "rp_sim": 0.85,
    "norm_method": "lognorm",
    "latent_dim": 64,
    "epochs": 1000,
    "n_critic": 5,
    "batch_size": 32,
    "d_model": 64,
    "num_heads": 4,
    "ff_dim": 256,
    "num_blocks": 3,
    "dropout": 0.1,
    "time2vec_dim": 16,
    "bg_lambda": 0.0,
    "ct_lambda": 0.0,
    # Adam del critico y del generador (los de V2-V6; Gulrajani et al. 2017 para las betas)
    "lr_d": 0.0001,
    "lr_g": 0.0002,
    "beta_1": 0.0,
    "beta_2": 0.9,
}


def ids(log: str, nombre: str) -> set:
    return set((COMUN / log / "split" / f"{nombre}_caseids.txt").read_text().split())


def _preparar_entrada(log: str, destino: Path) -> Path:
    """CSV en el formato que lee el entrenador (sin la columna de rol: el entrenador
    descubre los roles como lo hace el pipeline de Dynamics)."""
    import pandas as pd
    df = pd.read_csv(COMUN / log / "log.csv")
    for col in ("start_timestamp", "end_timestamp"):
        df[col] = (pd.to_datetime(df[col], utc=True).dt.tz_convert(None)
                   .dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / f"{log}.csv"
    df[["caseid", "task", "user", "start_timestamp", "end_timestamp"]].to_csv(salida, index=False)
    return salida


def entrenar(log: str, hiper: dict = None, semilla: int = 0, conjunto: str = "subtrain",
             excluir_regla: str = None, cada: int = 100, salida: Path = None) -> Path:
    """Entrena y devuelve la carpeta del modelo. `conjunto` = 'subtrain' (seleccion)
    o 'train' (train completo, sin validacion)."""
    import numpy as np
    import tensorflow as tf
    import GenerativeGAN.model_training.gan_trainer_v2 as modulo_v2
    from GenerativeGAN.model_training.gan_trainer_v2 import GANTrainerV2
    from support_modules import traces_evaluation as te

    hiper = {**HIPERPARAMETROS, **(hiper or {})}
    random.seed(semilla)
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

    train_ids, test_ids = ids(log, conjunto), ids(log, "test")
    regla_ini = COMUN / log / "reglas" / f"{excluir_regla}.ini" if excluir_regla else None

    class GANTrainerComun(GANTrainerV2):
        """GANTrainerV2 con la division comun y checkpoints."""

        def _split_timeline_70_10_20(self, rules_path, test_save_path=None):
            caso = self.log["caseid"].astype(str)
            self.log_train = self.log[caso.isin(train_ids)].copy()
            self.log_val = self.log.iloc[0:0].copy()
            self.log_test = self.log[caso.isin(test_ids)].copy()
            self.ocultos = []
            if regla_ini is not None:
                regla = te.extract_rules(path=str(regla_ini))
                cumple = self.log_train.groupby("caseid").apply(
                    lambda g: te.case_satisfies_rule(g, regla["path"], regla["rule"]))
                self.ocultos = sorted(map(str, cumple[cumple].index))
                self.log_train = self.log_train[~self.log_train["caseid"].astype(str)
                                                .isin(self.ocultos)].copy()
            self.n_train_cases = int(self.log_train["caseid"].nunique())
            self.n_test_cases = int(self.log_test["caseid"].nunique())
            self.pos_train_cases = self.train_prop = None   # se calcula al generar, por regla
            print(f"[RULE-GAN] Division comun ({conjunto}): {self.n_train_cases} train | "
                  f"{self.n_test_cases} test | {len(self.ocultos)} ocultos")

        def _train_wgan_gp(self, X, output_path):
            # El entrenador imprime su progreso cada 25 epocas desde dentro del bucle;
            # se intercepta ese print (solo en su modulo) para guardar los pesos del
            # generador sin copiar ni modificar el bucle de entrenamiento.
            carpeta_ckpt = Path(output_path) / "checkpoints"
            carpeta_ckpt.mkdir(parents=True, exist_ok=True)
            self.checkpoints = []
            entrenador = self

            def print_con_checkpoint(*args, **kwargs):
                builtins.print(*args, **kwargs)
                texto = args[0] if args else ""
                if isinstance(texto, str) and texto.startswith("[WGAN-GP] epoch "):
                    epoca = int(texto[len("[WGAN-GP] epoch "):].split("/")[0])
                    completadas = epoca + 1
                    if (epoca > 0 and epoca % cada == 0) or completadas == entrenador.epochs:
                        entrenador.generator.save_weights(
                            str(carpeta_ckpt / f"epoca_{completadas:04d}.h5"))
                        entrenador.checkpoints.append(completadas)

            modulo_v2.print = print_con_checkpoint
            try:
                super()._train_wgan_gp(X, output_path)
            finally:
                del modulo_v2.print
            if self.epochs not in self.checkpoints:
                raise RuntimeError("No se guardo el checkpoint final; revisar el formato del "
                                   "progreso que imprime GANTrainerV2.")

    class FeaturesSinConstantes(modulo_v2.FeaturesMannager):
        """scale_feature del entrenador, pero una variable constante queda en 0 en
        lugar de NaN (0/0 al normalizar). Pasa con la duracion en los logs de un solo
        timestamp (BPIC2012, Traffic Fines), que es siempre 0. El predictor ya trata
        ese caso (max == min). En RE y PE no cambia nada."""

        @staticmethod
        def scale_feature(log, feature, method, replace=False):
            log, args = original.scale_feature(log, feature, method, replace)
            if method in ("lognorm", "normal") and args["max_value"] == args["min_value"]:
                log[feature + "_norm"] = 0.0
            return log, args

    original = modulo_v2.FeaturesMannager
    entrada = _preparar_entrada(log, AQUI / "modelos" / log / "_entrada")
    base = salida or (AQUI / "modelos" / log)
    base.mkdir(parents=True, exist_ok=True)
    params = {
        **hiper,
        "file_name": f"{log}.csv",
        "model_type": ["transformer_wgan"],
        "read_options": {
            "timeformat": "%Y-%m-%d %H:%M:%S.%f",
            "column_names": {"Case ID": "caseid", "Activity": "task",
                             "lifecycle:transition": "event_type", "Resource": "user"},
            "one_timestamp": False,
        },
        "split_config": {"rules_path": "", "test_save_path": None},
    }
    antes = {p.name for p in base.iterdir() if p.is_dir()}
    inicio = time.time()
    modulo_v2.FeaturesMannager = FeaturesSinConstantes
    try:
        entrenador = GANTrainerComun(params, input_folder=str(entrada.parent), output_folder=str(base))
    finally:
        modulo_v2.FeaturesMannager = original
    nuevas = [p for p in base.iterdir() if p.is_dir() and p.name not in antes
              and not p.name.startswith("_")]
    carpeta = max(nuevas, key=lambda p: p.stat().st_mtime)
    meta = {
        "enfoque": "rule_gan", "log": log, "semilla": semilla, "conjunto": conjunto,
        "hiperparametros": hiper, "checkpoints": entrenador.checkpoints,
        "n_train": entrenador.n_train_cases, "n_test": entrenador.n_test_cases,
        "excluir_regla": excluir_regla, "casos_ocultos": entrenador.ocultos,
        "segundos": round(time.time() - inicio, 1),
    }
    (carpeta / "entrenamiento.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] Modelo RULE-GAN: {carpeta}")
    return carpeta


def preparar_entorno() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # el entrenador imprime caracteres no ASCII
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.chdir(RAIZ)                              # el entrenador usa rutas relativas al repo
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", default="{}", help="JSON con hiperparametros que reemplazan a los de V4")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--conjunto", default="subtrain", choices=["subtrain", "train"])
    ap.add_argument("--cada", type=int, default=100, help="epocas entre checkpoints")
    ap.add_argument("--excluir-regla", default=None,
                    help="hold-out: entrenar sin los casos que cumplen esta regla")
    args = ap.parse_args()
    preparar_entorno()
    entrenar(args.log, json.loads(args.config), args.semilla, args.conjunto,
             args.excluir_regla, args.cada)
