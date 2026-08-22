"""
Prepara los 2 archivos de configuracion de Simod que un log nuevo necesita antes de
correr el pipeline GENESIS (`docs/example/configuration.yaml` -> `configuration_original.yaml`
en `data/2.input_logs/<log>/` y `configuration_generated.yaml` en
`data/2.hallucination_logs/<log>/`).

Antes esto se hacia a mano (copiar el archivo dos veces y editar `train_log_path` uno
por uno) -- riesgo real de que las dos copias terminen divergiendo por error de
copy-paste. Este script copia la MISMA plantilla a los dos lugares, solo cambiando
`train_log_path` para que apunte al log correcto -- el resto de la configuracion
(discovery_type: undifferentiated, sin extraneous_activity_delays, etc., ver
CLAUDE.md 22.19-22.22) queda identica en las dos, como debe ser (CLAUDE.md 22.26: si
solo se corrige un lado, la comparacion de los 4 brazos queda desbalanceada por
metodologia, no por el efecto real de cada regla).

Uso (correr desde la raiz del repo):

    python -m rule_selection.setup_log_config --log RunningExample.csv

Sigue siendo necesario ajustar la plantilla a mano si el log tiene columnas o
convenciones distintas a `caseid/task/user/start_timestamp/end_timestamp` (ver
`log_ids` en el archivo generado) -- este script solo automatiza la copia con el
nombre correcto, no adapta el contenido a un log con estructura distinta.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _with_train_log_path(template_text: str, log_name: str) -> str:
    new_line = f"  train_log_path: ./{log_name}.csv.gz"
    updated, n = re.subn(r"^  train_log_path: .*$", new_line, template_text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError("No se encontro la linea 'train_log_path:' en la plantilla.")
    return updated


def setup_log_config(
    root: Path,
    log_filename: str,
    template_path: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    log_name = log_filename.replace(".csv", "")
    template_path = template_path or (root / "docs" / "example" / "configuration.yaml")
    if not template_path.exists():
        raise FileNotFoundError(f"No existe la plantilla {template_path}")

    template_text = template_path.read_text(encoding="utf-8")
    rendered = _with_train_log_path(template_text, log_name)

    asis_path = root / "data" / "2.input_logs" / log_name / "configuration_original.yaml"
    tobe_path = root / "data" / "2.hallucination_logs" / log_name / "configuration_generated.yaml"

    for out_path in (asis_path, tobe_path):
        if out_path.exists() and not force:
            print(f"[SKIP] {out_path} ya existe (usar --force para sobreescribir).")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        print(f"[OK] Escrito {out_path}")

    return asis_path, tobe_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.setup_log_config", description=__doc__)
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. RunningExample.csv)")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument(
        "--template",
        default=None,
        help="Plantilla a usar (default: docs/example/configuration.yaml)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Sobreescribir los archivos de configuracion si ya existen",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    template_path = Path(args.template).resolve() if args.template else None
    setup_log_config(root, args.log, template_path=template_path, force=args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
