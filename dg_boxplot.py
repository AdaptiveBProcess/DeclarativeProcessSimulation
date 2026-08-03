# -*- coding: utf-8 -*-
"""
dg_boxplot.py
Genera un panel 2x2 de boxplots comparando el baseline train_log contra
los 10 logs generados por el GAN, para las metricas RED, CTD, 2GD y CONF.

Metodologia (identica a Graziosi et al. 2024 — CVAE):
  - train_log : el log de entrenamiento se divide en N partes cronologicas
                de exactamente |test| trazas cada una. Cada parte se evalua
                vs el test split → N puntos en el boxplot (≤ 4).
  - GAN v4    : 10 logs sinteticos evaluados vs el test split → 10 puntos.
  - Referencia: test_split.csv (mismo para ambos enfoques).

Uso:
    python dg_boxplot.py

Busqueda automatica de datos GAN:
  1. data/4.simulation_results/RunningExample/metricas/metrics_runs_*.csv
  2. Si no existe: data/2.hallucination_logs/RunningExample/RunningExample_run*.csv
"""
import glob
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
NAME        = 'RunningExample'
METRICS_DIR = Path(f'data/4.simulation_results/{NAME}/metricas')
HALLUC_DIR  = Path(f'data/2.hallucination_logs/{NAME}')
TEST_SPLIT  = METRICS_DIR / 'test_split.csv'

# Log original (intentar preprocesado primero, luego original)
_LOG_CANDIDATES = [
    Path(f'data/0.logs/{NAME}_preprocessed/{NAME}.csv'),
    Path(f'data/0.logs/{NAME}/{NAME}.csv'),
]

# ── Paleta (dataviz skill — reference palette, light mode) ────────────────────
# Patron "emphasis": train_log = contexto gris, GAN = serie principal azul
_BLUE   = '#2a78d6'   # categorical slot 1 — GAN (serie principal)
_GRAY   = '#898781'   # muted ink — train_log (contexto/baseline)
_MEDIAN = '#0b0b0b'   # primary ink — linea de mediana
_GRID   = '#e1e0d9'   # gridline hairline
_SURF   = '#fcfcfb'   # chart surface
_TEXT_P = '#0b0b0b'   # primary text
_TEXT_S = '#52514e'   # secondary text


def _rgba(hex_color: str, alpha: float):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r, g, b, alpha)


# ── Definicion de metricas ─────────────────────────────────────────────────────
METRICS = [
    {
        'col':    'RED',
        'title':  'RED — Relative Event Distribution',
        'ylabel': 'EMD (posición relativa)',
        'better': 'lower',
        'fmt':    '{:.4f}',
    },
    {
        'col':    'CTD_horas',
        'title':  'CTD — Cycle Time Distribution',
        'ylabel': 'EMD (horas)',
        'better': 'lower',
        'fmt':    '{:.2f} h',
    },
    {
        'col':    '2GD',
        'title':  '2GD — 2-Gram Distance',
        'ylabel': 'EMD (bigramas)',
        'better': 'lower',
        'fmt':    '{:.4f}',
    },
    {
        'col':    'CONF',
        'title':  'CONF — Conformance Score',
        'ylabel': 'Trazas conformes (%)',
        'better': 'higher',
        'fmt':    '{:.1f} %',
    },
]


# ── Carga de metricas GAN ─────────────────────────────────────────────────────

def _latest_metrics_csv():
    files = sorted(glob.glob(str(METRICS_DIR / 'metrics_runs_*.csv')))
    return files[-1] if files else None


def _compute_gan_from_run_files() -> pd.DataFrame:
    run_files = sorted(HALLUC_DIR.glob(f'{NAME}_run*.csv'))
    if not run_files:
        print(f'[ERROR] No se encontraron logs en {HALLUC_DIR}')
        print('[ERROR] Ejecuta primero:  python dg_prediction.py')
        sys.exit(1)
    if not TEST_SPLIT.exists():
        print(f'[ERROR] test_split.csv no encontrado: {TEST_SPLIT}')
        print('[ERROR] Ejecuta primero:  python dg_training.py -m transformer_wgan')
        sys.exit(1)

    from evaluacion.evaluator import evaluate_batch
    print(f'[GAN] Calculando metricas para {len(run_files)} logs ...')
    df = evaluate_batch(
        ref_path=str(TEST_SPLIT),
        sim_paths=[str(p) for p in run_files],
        support_threshold=0.90,
    )
    df.insert(0, 'corrida', list(range(1, len(df) + 1)))
    return df


def load_gan_data() -> pd.DataFrame:
    csv = _latest_metrics_csv()
    if csv:
        print(f'[GAN] Cargando resultados precalculados:\n  {csv}')
        return pd.read_csv(csv)
    print('[GAN] metrics_runs_*.csv no encontrado — calculando desde cero ...')
    return _compute_gan_from_run_files()


# ── Baseline train_log ────────────────────────────────────────────────────────

def _load_original_log() -> pd.DataFrame:
    """Carga el log original (preprocesado o crudo) como DataFrame."""
    log_path = next((p for p in _LOG_CANDIDATES if p.exists()), None)
    if log_path is None:
        raise FileNotFoundError(
            f'Log original no encontrado. Rutas buscadas:\n'
            + '\n'.join(f'  {p}' for p in _LOG_CANDIDATES)
        )
    print(f'[TrainBaseline] Log original: {log_path}')
    df = pd.read_csv(log_path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # Normalizar nombres de columnas al formato esperado por el evaluador
    rename = {
        'case:concept:name': 'caseid',
        'concept:name':      'task',
        'org:resource':      'user',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ('start_timestamp', 'end_timestamp'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df.sort_values('start_timestamp').reset_index(drop=True)


def compute_train_baseline(n_parts: int = 4) -> pd.DataFrame:
    """
    Divide el log de entrenamiento (70% cronologico) en partes del tamano
    del test set y evalua cada parte vs el test split.

    Retorna DataFrame con las mismas columnas que evaluate_batch.
    """
    if not TEST_SPLIT.exists():
        print(f'[TrainBaseline] test_split.csv no encontrado: {TEST_SPLIT}')
        return pd.DataFrame()

    df_full = _load_original_log()

    # Recrear el split 70/10/20 por orden cronologico de casos
    all_cases  = df_full['caseid'].unique()          # ya ordenado por start_timestamp
    n_total    = len(all_cases)
    n_train    = int(n_total * 0.70)
    train_cases = all_cases[:n_train]
    df_train   = df_full[df_full['caseid'].isin(train_cases)].copy()

    # Tamano del test set
    test_df    = pd.read_csv(TEST_SPLIT, usecols=['caseid'])
    n_test     = test_df['caseid'].nunique()

    # Cuantas partes sin solapamiento caben
    max_parts   = len(train_cases) // n_test
    actual_parts = min(n_parts, max_parts)

    if actual_parts == 0:
        print(f'[TrainBaseline] Train ({len(train_cases)} trazas) demasiado '
              f'pequeno para partir en bloques de {n_test} trazas.')
        return pd.DataFrame()

    print(f'[TrainBaseline] {len(train_cases)} trazas train  |  '
          f'{n_test} trazas/parte  |  {actual_parts} partes')

    # Guardar cada parte como CSV temporal y evaluar en batch
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        part_paths = []
        for i in range(actual_parts):
            part_cases = train_cases[i * n_test : (i + 1) * n_test]
            part_df    = df_train[df_train['caseid'].isin(part_cases)]
            part_path  = tmp_dir / f'train_part_{i+1:02d}.csv'
            part_df.to_csv(part_path, index=False)
            part_paths.append(str(part_path))

        from evaluacion.evaluator import evaluate_batch
        print(f'[TrainBaseline] Evaluando {actual_parts} partes (MINERful 1 vez) ...')
        df_result = evaluate_batch(
            ref_path=str(TEST_SPLIT),
            sim_paths=part_paths,
            support_threshold=0.90,
        )
        df_result.insert(0, 'corrida', list(range(1, len(df_result) + 1)))
        return df_result

    finally:
        shutil.rmtree(tmp_dir)


# ── Visualizacion ─────────────────────────────────────────────────────────────

def _boxplot_series(ax, values, position, color, label, rng_seed):
    """Dibuja una serie (boxplot + jitter) en la posicion dada del eje."""
    bp = ax.boxplot(
        values,
        positions=[position],
        vert=True,
        patch_artist=True,
        widths=0.32,
        medianprops=dict(color=_MEDIAN, linewidth=2.5),
        whiskerprops=dict(color=color, linewidth=1.5, linestyle='-'),
        capprops=dict(color=color, linewidth=1.5),
        flierprops=dict(visible=False),
    )
    for patch in bp['boxes']:
        patch.set_facecolor(_rgba(color, 0.18))
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    # Puntos individuales con jitter
    n = len(values)
    jitter = np.random.default_rng(rng_seed).uniform(-0.10, 0.10, n)
    ax.scatter(
        np.full(n, position) + jitter,
        values,
        color=color,
        s=28,
        zorder=4,
        alpha=0.80,
        linewidths=0,
    )
    return bp


def _annotate_median(ax, values, position, meta):
    """Escribe el valor de la mediana a la derecha de la caja."""
    median_val = np.median(values)
    ax.text(
        position + 0.22, median_val,
        meta['fmt'].format(median_val),
        va='center', ha='left',
        fontsize=8, color=_TEXT_P, fontweight='semibold',
    )


def _style_ax(ax, meta, labels):
    """Aplica estilo minimalista y etiquetas del eje X."""
    ax.set_facecolor(_SURF)
    ax.set_title(meta['title'], fontsize=9.5, color=_TEXT_P,
                 pad=7, loc='left', fontweight='semibold')
    ax.set_ylabel(meta['ylabel'], fontsize=8, color=_TEXT_S, labelpad=5)

    positions = list(range(1, len(labels) + 1))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, color=_TEXT_S)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=8, labelcolor=_TEXT_S, length=0)

    ax.set_xlim(0.4, len(labels) + 0.7)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(_GRID)
    ax.spines['left'].set_linewidth(1)

    ax.yaxis.grid(True, color=_GRID, linewidth=1, linestyle='-')
    ax.set_axisbelow(True)

    arrow = '↓ mejor' if meta['better'] == 'lower' else '↑ mejor'
    ax.text(
        0.97, 0.03, arrow,
        transform=ax.transAxes,
        fontsize=7.5, color=_TEXT_S,
        ha='right', va='bottom', style='italic',
    )


def plot_boxplots(df_gan: pd.DataFrame, df_train: pd.DataFrame, output_path: Path):
    """Genera el panel 2x2 con train_log (gris) y GAN (azul) por subplot."""

    # Normalizar CONF a % si esta en [0,1]
    for df in (df_gan, df_train):
        if 'CONF' in df.columns and not df['CONF'].dropna().empty:
            if df['CONF'].dropna().max() <= 1.0:
                df['CONF'] = df['CONF'] * 100

    has_train = not df_train.empty
    labels    = (['train_log'] if has_train else []) + ['GAN v4']
    n_gan     = len(df_gan)
    n_train   = len(df_train) if has_train else 0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor(_SURF)

    for ax, meta in zip(axes.flatten(), METRICS):
        col = meta['col']

        gan_vals   = df_gan[col].dropna().tolist()   if col in df_gan.columns   else []
        train_vals = df_train[col].dropna().tolist() if (has_train and col in df_train.columns) else []

        if not gan_vals and not train_vals:
            ax.set_visible(False)
            continue

        pos = 1
        if train_vals:
            _boxplot_series(ax, train_vals, pos, _GRAY, 'train_log', rng_seed=0)
            _annotate_median(ax, train_vals, pos, meta)
            pos += 1

        if gan_vals:
            _boxplot_series(ax, gan_vals, pos, _BLUE, 'GAN v4', rng_seed=42)
            _annotate_median(ax, gan_vals, pos, meta)

        _style_ax(ax, meta, labels)

    # Titulo general
    subtitle = f'train_log: {n_train} partes   |   GAN v4: {n_gan} generaciones'
    fig.suptitle(
        f'Distribución de métricas — {NAME}\n{subtitle}',
        fontsize=11, color=_TEXT_P, y=1.02, fontweight='semibold',
    )

    # Leyenda compacta debajo del titulo
    handles = []
    if has_train:
        handles.append(plt.Rectangle((0, 0), 1, 1,
                                     fc=_rgba(_GRAY, 0.18), ec=_GRAY, lw=1.5,
                                     label='train_log (baseline)'))
    handles.append(plt.Rectangle((0, 0), 1, 1,
                                 fc=_rgba(_BLUE, 0.18), ec=_BLUE, lw=1.5,
                                 label='GAN v4'))
    fig.legend(handles=handles, loc='upper right', fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.98, 1.01))

    plt.tight_layout(pad=1.6, w_pad=3.0, h_pad=3.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor=_SURF)
    print(f'[Plot] Figura guardada: {output_path}')
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Metricas del GAN (10 corridas)
    df_gan = load_gan_data()

    # 2. Baseline train_log (N partes cronologicas del tamano del test)
    print()
    df_train = compute_train_baseline(n_parts=4)

    # 3. Resumen en consola
    metric_cols = [m['col'] for m in METRICS]
    for nombre, df in [('GAN v4', df_gan), ('train_log', df_train)]:
        cols = [c for c in metric_cols if c in df.columns]
        if cols and not df.empty:
            print(f'\n[Metrics] {nombre} ({len(df)} muestras):')
            print(df[cols].agg(['mean', 'std', 'min', 'max']).to_string())
    print()

    # 4. Visualizacion
    out_path = METRICS_DIR / f'boxplot_{NAME}.png'
    plot_boxplots(df_gan, df_train, out_path)


if __name__ == '__main__':
    main()
