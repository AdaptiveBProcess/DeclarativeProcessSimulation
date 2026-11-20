"""
Separa una validacion temporal dentro del train comun, para elegir modelos sin
tocar el test.

Igual que Dynamics (GenerativeLSTM/model_optimizer.py) y Camargo et al. (PeerJ CS
2021): el train se divide 80/20 en orden cronologico de inicio de caso; el 20%
mas reciente es la validacion.

Resultado, por log:
    experimentos/comun/<log>/split/subtrain_caseids.txt   (se entrena con esto)
    experimentos/comun/<log>/split/val_caseids.txt        (se elige con esto)
train_caseids.txt y test_caseids.txt no cambian.

Uso:
    python experimentos/comun/preparar_validacion.py --log RunningExample PurchasingExample
"""
import argparse
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent


def preparar(log: str, fraccion: float) -> None:
    split = AQUI / log / "split"
    train = (split / "train_caseids.txt").read_text().split()
    df = pd.read_csv(AQUI / log / "log.csv")
    df = df[df["caseid"].astype(str).isin(set(train))]
    inicio = (pd.to_datetime(df["start_timestamp"], utc=True)
              .groupby(df["caseid"].astype(str)).min().sort_values(kind="mergesort"))
    n_val = int(round(fraccion * len(inicio)))
    val = list(inicio.index[-n_val:])
    sub = list(inicio.index[:-n_val])
    (split / "val_caseids.txt").write_text("\n".join(sorted(val, key=str)) + "\n")
    (split / "subtrain_caseids.txt").write_text("\n".join(sorted(sub, key=str)) + "\n")
    print(f"[ok] {log}: {len(sub)} subtrain | {len(val)} validacion "
          f"(desde {inicio.iloc[-n_val]}) | {len(train)} train total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--fraccion", type=float, default=0.2)
    args = ap.parse_args()
    for log in args.log:
        preparar(log, args.fraccion)
