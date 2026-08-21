"""
Test de rule_selection.kpi contra un `_prosimos_stats.csv` real ya existente en el
repo (no requiere Docker/Simod/Prosimos -- el archivo ya fue generado antes).
"""

from pathlib import Path

import pytest

from rule_selection.kpi import analizar_metricas_archivo, promediar_iteraciones, mean_pce_over_replicas

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_STATS_CSV = (
    REPO_ROOT
    / "data"
    / "4.simulation_results"
    / "RunningExample"
    / "asis_20260314_134706"
    / "RunningExample_prosimos_stats.csv"
)


@pytest.mark.skipif(not SAMPLE_STATS_CSV.exists(), reason="Archivo de muestra no encontrado en el repo")
def test_analizar_metricas_archivo_parses_overall_statistics():
    df = analizar_metricas_archivo(SAMPLE_STATS_CSV)
    for kpi in ["cycle_time", "processing_time", "waiting_time"]:
        assert kpi in df.index
    assert df.loc["cycle_time", "Average"] > 0
    assert df.loc["processing_time", "Average"] > 0


@pytest.mark.skipif(not SAMPLE_STATS_CSV.exists(), reason="Archivo de muestra no encontrado en el repo")
def test_mean_pce_over_replicas_single_file():
    pce = mean_pce_over_replicas([SAMPLE_STATS_CSV])
    # PCE es un porcentaje: procesamiento / ciclo * 100 -- debe caer en un rango sano.
    assert 0 < pce <= 100


@pytest.mark.skipif(not SAMPLE_STATS_CSV.exists(), reason="Archivo de muestra no encontrado en el repo")
def test_promediar_iteraciones_two_replicas_same_file():
    resumen = promediar_iteraciones([SAMPLE_STATS_CSV, SAMPLE_STATS_CSV])
    # Con la misma replica dos veces, la desviacion estandar debe ser 0.
    assert resumen.loc["PCE (%)", "Desv. Estandar"] == pytest.approx(0.0, abs=1e-9)
