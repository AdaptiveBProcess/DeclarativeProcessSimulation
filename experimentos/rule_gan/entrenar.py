"""
Entrena RULE-GAN (GANTrainerV2, Transformer + WGAN-GP) con la division comun.

Usa los mismos hiperparametros que dg_training.py -m transformer_wgan. La unica
diferencia es la division: en vez del 70/10/20 propio del entrenador, se usan los
casos de experimentos/comun/<log>/split/ (los mismos que Dynamics), para que los
tres enfoques se entrenen y evaluen sobre los mismos casos.

El entrenamiento no depende de la regla: un modelo por log sirve para todas las
reglas (salvo en el hold-out, donde hay que entrenar sin las trazas ocultas; ver
--excluir-regla).

Uso:
    python experimentos/rule_gan/entrenar.py --log RunningExample [--epocas 1000]
    python experimentos/rule_gan/entrenar.py --log RunningExample --excluir-regla r1_directly_TaskB_TaskC

Resultado: experimentos/rule_gan/modelos/<log>/<AAAAMMDD_HHMMSS>/
    gen/                         generador (SavedModel)
    parameters/model_parameters.json, test_log.csv, <log>_ASIS.csv
    entrenamiento.json           parametros y casos usados
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
COMUN = RAIZ / "experimentos" / "comun"

# Configuracion de dg_training.py para transformer_wgan (v4 CPU)
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
}


def _ids(log: str, nombre: str) -> set:
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


def entrenar(log: str, epocas: int, semilla: int, excluir_regla: str = None) -> None:
    import numpy as np
    import tensorflow as tf
    from GenerativeGAN.model_training.gan_trainer_v2 import GANTrainerV2
    from support_modules import traces_evaluation as te

    random.seed(semilla)
    np.random.seed(semilla)
    tf.random.set_seed(semilla)

    train_ids, test_ids = _ids(log, "train"), _ids(log, "test")
    regla_ini = None
    if excluir_regla:
        regla_ini = COMUN / log / "reglas" / f"{excluir_regla}.ini"

    class GANTrainerComun(GANTrainerV2):
        """GANTrainerV2 con la division comun en lugar de su 70/10/20."""

        def _split_timeline_70_10_20(self, rules_path, test_save_path=None):
            ids = self.log["caseid"].astype(str)
            self.log_train = self.log[ids.isin(train_ids)].copy()
            self.log_val = self.log.iloc[0:0].copy()
            self.log_test = self.log[ids.isin(test_ids)].copy()
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
            # La proporcion de la regla se calcula al generar, para cada regla.
            self.pos_train_cases = self.train_prop = None
            print(f"[RULE-GAN] Division comun: {self.n_train_cases} train | "
                  f"{self.n_test_cases} test | {len(self.ocultos)} ocultos (hold-out)")

    entrada = _preparar_entrada(log, AQUI / "modelos" / log / "_entrada")
    salida = AQUI / "modelos" / log
    params = {
        **HIPERPARAMETROS,
        "epochs": epocas,
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
    antes = {p.name for p in salida.iterdir() if p.is_dir()}
    entrenador = GANTrainerComun(params, input_folder=str(entrada.parent), output_folder=str(salida))
    nuevas = [p for p in salida.iterdir() if p.is_dir() and p.name not in antes and p.name != "_entrada"]
    carpeta = max(nuevas, key=lambda p: p.stat().st_mtime)
    meta = {
        "enfoque": "rule_gan", "log": log, "semilla": semilla, "epocas": epocas,
        "hiperparametros": {**HIPERPARAMETROS, "epochs": epocas},
        "n_train": entrenador.n_train_cases, "n_test": entrenador.n_test_cases,
        "excluir_regla": excluir_regla, "casos_ocultos": entrenador.ocultos,
    }
    (carpeta / "entrenamiento.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] Modelo RULE-GAN: {carpeta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--epocas", type=int, default=HIPERPARAMETROS["epochs"])
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--excluir-regla", default=None,
                    help="hold-out: entrenar sin los casos de train que cumplen esta regla")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")   # el entrenador imprime caracteres no ASCII
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.chdir(RAIZ)
    sys.path.insert(0, str(RAIZ))
    entrenar(args.log, args.epocas, args.semilla, args.excluir_regla)
