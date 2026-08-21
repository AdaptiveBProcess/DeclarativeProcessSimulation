"""Utilidades internas compartidas dentro de rule_selection/."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def chdir(path: Path):
    """
    Cambia el cwd temporalmente y lo restaura al salir. Varios scripts legacy
    reutilizados por este paquete (`ejecutar_minerful`, `dg_training.py::main`)
    asumen rutas relativas al cwd -- se aisla ese requisito aca en vez de
    esparcirlo por cada modulo que los llama.
    """
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
