"""
Diagnostico de calendarios de recursos en un modelo BPS descubierto por Simod
(CLAUDE.md 22.19-22.20) -- util para comparar que tan fragmentado queda el
calendario de cada recurso bajo distintas configuraciones de descubrimiento
(ej. `discovery_type: differentiated` vs `undifferentiated`).

No requiere Docker ni Simod en ejecucion -- solo lee el `<name>.json` que Simod ya
dejo en disco tras un descubrimiento.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _parse_time(value: str) -> datetime:
    """Simod a veces escribe segundos con fraccion (ej. '23:59:59.999000',
    visto en PurchasingExample -- nunca en RunningExample) y a veces sin ella
    ('17:00:00') -- se prueban ambos formatos en vez de asumir uno solo."""
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Formato de hora no reconocido: {value!r}")


def _period_hours(period: dict) -> float:
    begin = _parse_time(period["beginTime"])
    end = _parse_time(period["endTime"])
    return (end - begin).total_seconds() / 3600


def summarize_resource_calendars(bps_model_json_path: Path) -> pd.DataFrame:
    """
    Lee `resource_calendars` de un modelo BPS de Simod (`<name>.json`) y devuelve,
    por calendario, las horas/semana disponibles y el numero de bloques.

    Referencia: un calendario laboral completo Lunes-Viernes 9-17 son 40 h/semana en
    5 bloques. Con `discovery_type: differentiated` sobre `RunningExample` se
    encontraron 31 calendarios con una media de ~15.8 h/semana (39.6% de una semana
    completa) en bloques sueltos de 1 hora -- ver CLAUDE.md 22.20 para el analisis
    completo y la hipotesis de que `undifferentiated` lo resuelve.
    """
    with open(bps_model_json_path) as f:
        model = json.load(f)

    rows = []
    for cal in model.get("resource_calendars", []):
        periods = cal.get("time_periods", [])
        rows.append(
            {
                "calendar_id": cal["id"],
                "weekly_hours": sum(_period_hours(p) for p in periods),
                "n_blocks": len(periods),
            }
        )
    return pd.DataFrame(rows)


def find_bps_model_json(bps_folder: Path, model_name: str) -> Path | None:
    """
    Localiza `<model_name>.json` dentro de la carpeta que Simod descubrio (busca en
    `<bps_folder>/*/best_result/<model_name>.json`, la estructura que usa Simod
    dentro de su carpeta de salida con timestamp).
    """
    matches = list(bps_folder.glob(f"*/best_result/{model_name}.json"))
    return matches[0] if matches else None


_VIRTUAL_RESOURCE_ID = "ALWAYS_AVAILABLE"
_VIRTUAL_CALENDAR_ID = "ALWAYS_AVAILABLE_calendar"
_DAYS_OF_WEEK = [
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
]


def _is_instantaneous_task(entry: dict) -> bool:
    """Una tarea es 'instantanea' si TODOS los recursos elegibles tienen una
    distribucion de duracion fija en 0 -- ej. marcadores sinteticos como
    `EVENT 1 START`/`EVENT 7 END`, que en el log real no tienen `org:resource`
    asignado (CLAUDE.md 22.24), a diferencia de tareas reales con duracion real."""
    resources = entry.get("resources", [])
    if not resources:
        return False
    return all(
        r.get("distribution_name") == "fix" and r.get("distribution_params", [{}])[0].get("value") == 0.0
        for r in resources
    )


def neutralize_instantaneous_task_resources(
    bps_model_json_path: Path,
    output_path: Path | None = None,
    virtual_amount: int = 999999,
) -> list[str]:
    """
    Post-procesa un modelo BPS ya descubierto por Simod (CLAUDE.md 22.24-22.27):
    detecta tareas "instantaneas" (duracion fija en 0 para TODOS sus recursos
    elegibles) y las reasigna a un recurso virtual `ALWAYS_AVAILABLE` con calendario
    24/7 y capacidad muy alta, en vez de dejarlas competir por el calendario de los
    recursos reales -- evita el atasco de fin de semana encontrado en §22.24 (un
    marcador sin trabajo real que quedaba esperando el calendario de un recurso
    individual, ~93% del gap de PCE que quedaba tras corregir calendarios y demoras
    extrañas).

    Modifica en el sitio (`output_path=None`) o escribe en `output_path` si se pasa.
    Devuelve la lista de `task_id` neutralizados (vacia si no se encontro ninguna
    tarea instantanea -- no es un error, simplemente no hay nada que neutralizar).

    No modifica el simulador (Prosimos) ni el generador LSTM -- solo el JSON de
    parametros de simulacion que Simod ya dejo en disco, antes de que Prosimos lo
    lea para simular.
    """
    with open(bps_model_json_path) as f:
        model = json.load(f)

    task_resource_distribution = model.get("task_resource_distribution", [])
    instantaneous_ids = [t["task_id"] for t in task_resource_distribution if _is_instantaneous_task(t)]

    if not instantaneous_ids:
        return []

    # 1. Calendario 24/7 para el recurso virtual.
    virtual_calendar = {
        "id": _VIRTUAL_CALENDAR_ID,
        "name": _VIRTUAL_CALENDAR_ID,
        "time_periods": [
            {"from": day, "to": day, "beginTime": "00:00:00", "endTime": "23:59:59"} for day in _DAYS_OF_WEEK
        ],
    }
    model.setdefault("resource_calendars", []).append(virtual_calendar)

    # 2. Recurso virtual, agregado al primer resource_profile (mismo patron que usa
    # Simod para `undifferentiated`: un solo grupo con todos los recursos elegibles).
    virtual_resource = {
        "id": _VIRTUAL_RESOURCE_ID,
        "name": _VIRTUAL_RESOURCE_ID,
        "amount": virtual_amount,
        "cost_per_hour": 0,
        "calendar": _VIRTUAL_CALENDAR_ID,
        "assignedTasks": list(instantaneous_ids),
    }
    resource_profiles = model.get("resource_profiles", [])
    if resource_profiles:
        resource_profiles[0].setdefault("resource_list", []).append(virtual_resource)
    else:
        resource_profiles.append(
            {"id": "ALWAYS_AVAILABLE_profile", "name": "ALWAYS_AVAILABLE_profile", "resource_list": [virtual_resource]}
        )
        model["resource_profiles"] = resource_profiles

    # 3. Quitar las tareas instantaneas de las listas assignedTasks de los recursos
    # reales -- ya no deben competir por ellas.
    for profile in resource_profiles:
        for resource in profile.get("resource_list", []):
            if resource["id"] == _VIRTUAL_RESOURCE_ID:
                continue
            resource["assignedTasks"] = [t for t in resource.get("assignedTasks", []) if t not in instantaneous_ids]

    # 4. En task_resource_distribution, reemplazar la lista de recursos elegibles de
    # cada tarea instantanea por SOLO el recurso virtual (misma distribucion fix/0.0
    # ya descubierta -- no se toca la duracion, solo quien la "ejecuta").
    for entry in task_resource_distribution:
        if entry["task_id"] in instantaneous_ids:
            entry["resources"] = [
                {"resource_id": _VIRTUAL_RESOURCE_ID, "distribution_name": "fix", "distribution_params": [{"value": 0.0}]}
            ]

    target_path = output_path or bps_model_json_path
    with open(target_path, "w") as f:
        json.dump(model, f, indent=2)

    return instantaneous_ids
