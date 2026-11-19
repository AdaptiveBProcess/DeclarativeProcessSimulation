"""
Arma el log y la division comunes a los tres enfoques para los logs de Dynamics.

Fuente: los archivos que Dynamics publico junto a cada modelo (descargados con
experimentos/dynamics/descargar_paquete.py):
    parameters/<log>_ASIS.csv  -> log completo usado por Dynamics (sin los eventos
                                  artificiales de inicio/fin) con el rol de cada recurso
    parameters/test_log.csv    -> casos de test de Dynamics (los ultimos del log)

Resultado, por log:
    experimentos/comun/<log>/log.csv            caseid, task, user, role, start_timestamp, end_timestamp
    experimentos/comun/<log>/split/test_caseids.txt
    experimentos/comun/<log>/split/train_caseids.txt

Asi RULE-GAN y CVAE se entrenan con los mismos casos que Dynamics y los tres se
evaluan sobre el mismo test.

Uso:
    python experimentos/comun/preparar_logs_dynamics.py
"""
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
MODELOS = AQUI.parent / "dynamics" / "modelos"
COLUMNAS = ["caseid", "task", "user", "role", "start_timestamp", "end_timestamp"]


def preparar(log: str, carpeta_modelo: Path) -> None:
    param = carpeta_modelo / "parameters"
    asis = pd.read_csv(param / f"{log}_ASIS.csv")
    test = pd.read_csv(param / "test_log.csv")

    destino = AQUI / log
    (destino / "split").mkdir(parents=True, exist_ok=True)
    asis = asis[COLUMNAS].sort_values(["caseid", "start_timestamp"], kind="mergesort")
    asis.to_csv(destino / "log.csv", index=False)

    casos = sorted(asis["caseid"].unique())
    test_ids = sorted(test["caseid"].unique())
    faltan = set(test_ids) - set(casos)
    if faltan:
        raise ValueError(f"{log}: {len(faltan)} casos de test no estan en el log AS-IS")
    train_ids = [c for c in casos if c not in set(test_ids)]
    (destino / "split" / "test_caseids.txt").write_text("\n".join(map(str, test_ids)) + "\n")
    (destino / "split" / "train_caseids.txt").write_text("\n".join(map(str, train_ids)) + "\n")
    print(f"[ok] {log}: {len(casos)} casos ({len(train_ids)} train, {len(test_ids)} test), "
          f"{asis['task'].nunique()} actividades, {asis['role'].nunique()} roles -> {destino}")


if __name__ == "__main__":
    for carpeta_log in sorted(p for p in MODELOS.iterdir() if p.is_dir()):
        for carpeta_modelo in sorted(p for p in carpeta_log.iterdir() if p.is_dir()):
            preparar(carpeta_log.name, carpeta_modelo)
