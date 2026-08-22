"""
Parseo de resultados de simulacion Prosimos (`*_prosimos_stats.csv`) y agregacion
entre replicas.

Originalmente extraido de `validacion_resultados_2.py` (las funciones puras, sin
efectos secundarios) para poder reutilizarlo desde el pipeline nuevo sin arrastrar
el codigo a nivel de modulo de ese script (que corria analisis sobre rutas
absolutas hardcodeadas apenas se importaba). Ese script quedo con su propia copia
de estas funciones en vez de importarlas de vuelta, y sus rutas hardcodeadas
apuntaban a una recoleccion de datos de mayo de 2026 que ya no existe en disco
-- confirmado no funcional (fallaba con FileNotFoundError apenas se ejecutaba) y
borrado del repo.
"""

from __future__ import annotations

import io

import pandas as pd


def analizar_metricas_archivo(file_path) -> pd.DataFrame:
    """
    Lee un archivo `_prosimos_stats.csv` y devuelve el bloque "Overall Scenario
    Statistics" como DataFrame indexado por `KPI` (columnas Min/Max/Average/
    Accumulated Value/Trace Ocurrences).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    start_index = -1
    for i, line in enumerate(lines):
        if "Overall Scenario Statistics" in line:
            start_index = i + 1
            break

    if start_index == -1:
        raise ValueError(f"No se encontro 'Overall Scenario Statistics' en {file_path}")

    csv_content = "".join([l for l in lines[start_index:] if l.strip()])
    column_names = ["KPI", "Min", "Max", "Average", "Accumulated Value", "Trace Ocurrences"]

    df = pd.read_csv(io.StringIO(csv_content), names=column_names, header=0)
    df["KPI"] = df["KPI"].str.strip()
    df.set_index("KPI", inplace=True)
    return df


def promediar_iteraciones(rutas_archivos: list) -> pd.DataFrame | None:
    """
    Itera sobre los `_prosimos_stats.csv` de varias replicas y calcula promedio y
    desviacion estandar de cycle_time/processing_time/waiting_time/idle_time, mas
    PCE (%) e IL (%) derivados.
    """
    if not rutas_archivos:
        return None

    resultados = []
    for ruta in rutas_archivos:
        df = analizar_metricas_archivo(ruta)
        try:
            metrics = {
                "cycle_time": df.loc["cycle_time", "Average"],
                "processing_time": df.loc["processing_time", "Average"],
                "waiting_time": df.loc["waiting_time", "Average"],
            }
            metrics["idle_time"] = (
                df.loc["idle_time", "Average"] if "idle_time" in df.index else 0
            )
            metrics["PCE (%)"] = (metrics["processing_time"] / metrics["cycle_time"]) * 100
            metrics["IL (%)"] = (metrics["waiting_time"] / metrics["cycle_time"]) * 100
            resultados.append(metrics)
        except KeyError as e:
            print(f"Error extrayendo metricas en {ruta}: {e}")

    df_resultados = pd.DataFrame(resultados)
    df_promedios = df_resultados.mean().to_frame(name="Promedio")
    df_promedios["Desv. Estandar"] = df_resultados.std()
    return df_promedios


def formatear_resumen(df_promedios: pd.DataFrame | None) -> dict:
    """Convierte los resultados de `promediar_iteraciones` al formato 'Media ± Desv'."""
    if df_promedios is None:
        return {}

    resultados_str = {}
    for metrica in df_promedios.index:
        media = df_promedios.loc[metrica, "Promedio"]
        desv = df_promedios.loc[metrica, "Desv. Estandar"]
        if pd.isna(desv):
            desv = 0.0
        if metrica in ["PCE (%)", "IL (%)"]:
            resultados_str[metrica] = f"{media:.2f}% ± {desv:.2f}%"
        else:
            resultados_str[metrica] = f"{media:.2f} ± {desv:.2f}"
    return resultados_str


def mean_pce_over_replicas(stats_csv_paths: list) -> float:
    """
    PCE promedio (0-100) entre replicas de un mismo brazo/candidata -- usado para
    elegir la ganadora del screening (CLAUDE.md 12.5, criterio confirmado: media
    simple de PCE simulado, no una segunda aplicacion de la formula de Welch).
    """
    resumen = promediar_iteraciones(stats_csv_paths)
    if resumen is None:
        raise ValueError("No se recibieron rutas de _prosimos_stats.csv")
    return float(resumen.loc["PCE (%)", "Promedio"])
