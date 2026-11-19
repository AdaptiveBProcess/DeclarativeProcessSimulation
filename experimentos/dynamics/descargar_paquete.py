"""
Descarga el paquete de reproducibilidad de Dynamics (Baron-Espitia et al., 2024)
desde Zenodo y extrae los modelos LSTM publicados.

    Zenodo: https://doi.org/10.5281/zenodo.10475754 (version 11067569)

Resultado:
    experimentos/dynamics/paquete_zenodo/reproducibility_package.zip
    experimentos/dynamics/modelos/<log>/<id_modelo>/   (h5 + parameters/)

Uso (desde cualquier carpeta):
    python experimentos/dynamics/descargar_paquete.py
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
URL = "https://zenodo.org/api/records/11067569/files/reproducibility_package.zip/content"
MD5 = "56e7b2353ac6312be580a6c7f519e9c0"
ZIP = AQUI / "paquete_zenodo" / "reproducibility_package.zip"
PREFIJO = "data_simulationModels/simulationModels_what-if_generated/"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar() -> None:
    if ZIP.exists() and _md5(ZIP) == MD5:
        print(f"[ok] Paquete ya descargado: {ZIP}")
        return
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL} ...")
    urllib.request.urlretrieve(URL, ZIP)
    if _md5(ZIP) != MD5:
        sys.exit(f"[error] MD5 distinto al publicado en Zenodo ({MD5}); revisar la descarga.")
    print(f"[ok] Descargado y verificado: {ZIP}")


def extraer_modelos() -> None:
    """Un modelo por log: la carpeta que contiene <log>.h5. Se omiten los SE_*.csv
    (resultados de sus corridas) para que la carpeta quede como el modelo publicado."""
    z = zipfile.ZipFile(ZIP)
    carpetas = {}
    for n in z.namelist():
        if n.startswith(PREFIJO) and n.endswith(".h5") and n.count("/") == 3:
            carpetas[n.split("/")[2]] = Path(n).stem
    for id_modelo, log in sorted(carpetas.items()):
        destino = AQUI / "modelos" / log / id_modelo
        for n in z.namelist():
            pre = f"{PREFIJO}{id_modelo}/"
            if not n.startswith(pre) or n.endswith("/") or "/SE_" in n:
                continue
            p = destino / n[len(pre):]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(n))
        print(f"[ok] Modelo {log}: {destino}")


if __name__ == "__main__":
    descargar()
    extraer_modelos()
