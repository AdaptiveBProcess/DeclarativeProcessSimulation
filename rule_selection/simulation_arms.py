"""
Runner reutilizable: alucinacion LSTM -> descubrimiento Simod -> simulacion Prosimos
(CLAUDE.md Seccion 12.5/12.6).

No modifica `dg_prediction.py` ni `params/Params.py` -- reutiliza sus funciones/
dataclass tal cual, pero les pasa carpetas de salida namespaced por corrida
(`run_label`) en vez de las rutas compartidas que usa `dg_prediction.py::main()`,
porque este pipeline corre hasta ~8 descubrimientos de Simod por log (5 candidatas
del screening + Placebo + Top-Support + extra de GENESIS) y depender de "la carpeta
mas reciente por fecha de modificacion" (`pa.get_latest_output_folder`) seria fragil
con tantas corridas encima de la misma carpeta compartida.

No repite los bugs ya identificados en `dg_prediction.py::main()`: sin la
re-simulacion redundante que sobrescribe la replica 1 con un `total_cases` distinto,
y con nombres de carpeta de replica explicitos (`{run_label}_rep{i}`) en vez del
truco de `datetime.now() + timedelta(seconds=N)`.

Confirmado leyendo `GenerativeLSTM/model_prediction/model_predictor.py`:
`extract_rules(rules_path)` se llama INCONDICIONALMENTE, incluso para
`variant="Random Choice"` (el Placebo) -- asi que todo brazo, incluido el Placebo,
necesita un `.ini` sintacticamente valido (aunque su contenido no filtre nada
durante la generacion sin restriccion).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import dg_prediction as dgp
from params.Params import Params
from rule_selection.calendar_diagnostics import find_bps_model_json, neutralize_instantaneous_task_resources
from rule_selection.candidates import write_rule_ini
from rule_selection.kpi import mean_pce_over_replicas
from support_modules import predictor_adapter as pa


@dataclass
class ArmRunHandle:
    """Resultado de correr un brazo (candidata/AS-IS/Placebo/Top-Support/GENESIS)."""

    label: str
    params: Params
    hallucinated_folder: Path | None = None
    bps_folder: Path | None = None  # carpeta namespaced que recibio el descubrimiento Simod
    resources_filename: str | None = None  # nombre del JSON de recursos a usar en simulate_model
    stats_csv_paths: list[Path] = field(default_factory=list)

    @property
    def mean_pce(self) -> float:
        return mean_pce_over_replicas(self.stats_csv_paths)


def _clear_stale_replicas(simulation_dir: Path, run_label: str) -> None:
    """
    Borra carpetas de replicas ya simuladas (`{run_label}_rep*`) antes de una nueva
    alucinacion/descubrimiento para ese `run_label`.

    Sin esto, `reconstruct_tobe_handle` (llamado desde un proceso CLI separado, ej.
    `run_final_comparison_cli.py`) puede encontrar replicas huerfanas de un
    descubrimiento ANTERIOR (con un modelo de recursos fusionado distinto) y
    mezclarlas con las nuevas sin darse cuenta -- contamina el promedio de PCE
    reportado. Bug real encontrado en PurchasingExample (CLAUDE.md 24.8): al
    re-descubrir `screening_..._Choose_best_option` con el AS-IS ya corregido, las 5
    replicas nuevas (rep1-5, PCE~96%) quedaron en disco junto a 15 replicas huerfanas
    de una corrida anterior con AS-IS roto (rep6-20, PCE~20-30%) -- `reconstruct_tobe_handle`
    encontro las 20 y las promedio todas juntas como si fueran validas.
    """
    for stale in simulation_dir.glob(f"{run_label}_rep*"):
        if stale.is_dir():
            shutil.rmtree(stale)


def _clear_shared_traces_generated(params: Params) -> None:
    """
    `GenerativeLSTM/model_prediction/model_predictor.py::export_predictions` only
    deletes the trace files under `traces_generated` AFTER a hallucination run
    finishes successfully (it is a scratch folder namespaced by the trained MODEL,
    not by our `run_label` -- every candidate/arm for a log reuses the same trained
    model, so they all resolve to the same `traces_generated` path). If a previous
    hallucination crashed mid-generation (e.g. the OOM in CLAUDE.md 25.4), its
    partial trace files are never cleaned up and sit there for the next run to pick
    up -- `get_stats_log_traces` scans the whole folder unconditionally, and case
    filenames (`gen-Case000.csv`, ...) collide across runs because
    `hallucination_cases` is fixed, so leftover files can be silently miscounted as
    part of a later, unrelated run's output. Clear it defensively before every
    fresh hallucination.
    """
    models_root = Path(params.routes["models"])
    if not models_root.exists():
        return
    latest = pa.get_latest_output_folder(str(models_root))
    if latest is None:
        return
    traces_dir = models_root / latest / "parameters" / "traces_generated"
    if traces_dir.exists():
        shutil.rmtree(traces_dir)


def make_candidate_params(
    base: Params,
    candidate_id: str,
    template: str,
    arg0: str,
    arg1: str | None,
    variant: str = "Rules Based Random Choice",
) -> Params:
    """
    Crea un `Params` propio para una candidata: mismo log/root que `base`, pero con
    un `rules_filename` propio -- y escribe ese `.ini` ya mismo (via
    `candidates.write_rule_ini`, que no tiene el bug de rutas de
    `convertir_declare_to_declarative`).
    """
    candidate_params = Params(
        root=base.root,
        log_filename=base.log_filename,
        # `rep` es el numero de veces que el LSTM regenera/revalida internamente su
        # propia alucinacion (GenerativeLSTM/model_prediction/model_predictor.py,
        # confirmado por exploracion) -- NO es el numero de replicas de Prosimos que
        # nos interesa para el screening/comparacion final. Se deja en 1.
        rep=1,
        variant=variant,
        rules_filename=f"{candidate_id}.ini",
        tobe_cases=base.tobe_cases,
        # BUG REAL encontrado y corregido (CLAUDE.md 27.9): faltaba reenviar
        # hallucination_cases -- con el default None, Params.simulation cae a
        # total_cases (10469 para BPI_Challenge_2012) en vez del n=371 de Cochran
        # pedido explicitamente por CLI/orchestrate, para CADA candidata construida
        # por esta funcion (las 5 del screening, Top-Support y Placebo). Explica la
        # sobregeneracion masiva observada (Top-Support 1819 trazas, Placebo 3506)
        # -- no apuntaban a 371 en absoluto.
        hallucination_cases=base.hallucination_cases,
    )
    write_rule_ini(template, arg0, arg1, candidate_params.routes["rules"])
    return candidate_params


def discover_asis_bps(
    params: Params,
    run_label: str = "asis_shared",
    configuration_filename: str = "configuration_original.yaml",
    neutralize_instantaneous_resources: bool = False,
) -> Path:
    """
    Descubre el modelo Simod de AS-IS UNA sola vez para todo el log -- no depende de
    ninguna regla candidata (confirmado: `generate_bps_model(r["input"], ...)` solo
    usa el log original preprocesado), asi que se calcula una vez y se reutiliza
    tanto en el screening (12.5) como en los 4 brazos de la comparacion final (12.6).

    `configuration_filename` debe existir en `params.routes["input"]`
    (`data/2.input_logs/<log>/`) -- por defecto `configuration_original.yaml`, pero
    se puede pasar una variante (ej. `configuration_original_undifferentiated.yaml`)
    para probar otros hiperparametros de descubrimiento de Simod (ej. calendarios de
    recursos, ver CLAUDE.md 22.20) sin tocar el archivo original ni el resto del
    pipeline.

    `neutralize_instantaneous_resources=True` aplica
    `calendar_diagnostics.neutralize_instantaneous_task_resources` sobre el modelo
    recien descubierto -- corrige el atasco de fin de semana de marcadores sinteticos
    (ej. `EVENT 1 START`/`EVENT 7 END`) con recurso inventado (CLAUDE.md 22.24/22.27).
    """
    r = params.routes
    _clear_stale_replicas(r["simulation"], run_label)
    bps_asis_folder = r["bps_asis"] / run_label
    dgp.preprocess_xes_log(r["log"] / params.log_filename, output_folder=r["input"])
    dgp.generate_bps_model(r["input"], bps_asis_folder, configuration_filename)
    if neutralize_instantaneous_resources:
        model_json = find_bps_model_json(bps_asis_folder, params.name)
        if model_json is not None:
            neutralize_instantaneous_task_resources(model_json)
    return bps_asis_folder


def run_asis_arm(
    params: Params,
    bps_asis_folder: Path,
    replicas: int,
    run_label: str = "asis",
) -> ArmRunHandle:
    """AS-IS: sin alucinacion, sin descubrimiento nuevo -- solo `replicas` corridas de Prosimos."""
    r = params.routes
    handle = ArmRunHandle(
        label=run_label,
        params=params,
        bps_folder=bps_asis_folder,
        resources_filename=f"{params.name}.json",
    )

    for i in range(1, replicas + 1):
        rep_output = r["simulation"] / f"{run_label}_rep{i}"
        dgp.simulate_model(
            input_path=bps_asis_folder,
            output_path=rep_output,
            bpmn_filename=r["bpmn"],
            resources_filename=handle.resources_filename,
            total_cases=params.effective_tobe_cases,
        )
        handle.stats_csv_paths.append(rep_output / f"{params.name}_prosimos_stats.csv")

    return handle


def run_tobe_arm(
    params: Params,
    replicas: int,
    run_label: str,
    root_path: Path,
    bps_asis_folder: Path | None = None,
    discover: bool = True,
    existing_handle: ArmRunHandle | None = None,
    configuration_filename: str = "configuration_generated.yaml",
    neutralize_instantaneous_resources: bool = False,
) -> ArmRunHandle:
    """
    Un brazo TO-BE (Placebo / Top-Support / GENESIS / cada candidata del screening):
    alucinacion LSTM (filtrada por `params.rules_filename`/`params.variant`) +
    descubrimiento Simod, UNA vez, mas `replicas` corridas de Prosimos.

    Si `discover=False`, reutiliza el BPS ya descubierto en `existing_handle` en vez
    de volver a alucinar/descubrir -- asi es como el brazo GENESIS final reutiliza
    las 5 replicas ya corridas en el screening, sumando solo las que faltan para
    llegar a `R_final` (CLAUDE.md 12.5/12.7).

    `configuration_filename` debe existir en `params.routes["hallucinated"]`
    (`data/2.hallucination_logs/<log>/`, la carpeta compartida sin namespacing) --
    mismo mecanismo que `discover_asis_bps`, para poder probar variantes de
    configuracion de Simod (ver CLAUDE.md 22.20-22.26) en el lado TO-BE tambien.

    `neutralize_instantaneous_resources=True` aplica
    `calendar_diagnostics.neutralize_instantaneous_task_resources` sobre el JSON
    FUSIONADO (`adapt_resources`, el que realmente usa Prosimos via
    `handle.resources_filename`) -- mismo fix que en `discover_asis_bps`, aplicado
    despues de la fusion porque es el archivo final que se simula (CLAUDE.md 22.27).
    """
    r = params.routes
    sim = params.simulation

    if discover:
        if bps_asis_folder is None:
            raise ValueError("discover=True requiere bps_asis_folder (ver discover_asis_bps).")
        _clear_stale_replicas(r["simulation"], run_label)
        _clear_shared_traces_generated(params)

        hallucinated_folder = r["hallucinated"] / run_label
        dgp.call_predict(sim, r["models"], hallucinated_folder, r["rules"], root_path)
        dgp.preprocess_xes_log(
            hallucinated_folder / params.log_filename,
            output_folder=hallucinated_folder,
            filter_sentinels=True,
        )

        # Simod busca la config en la MISMA carpeta que el log de entrada que se
        # monta como /usr/src/Simod/resources. Ese archivo es estatico por log (uno
        # por carpeta de log, no se genera por codigo -- vive en la carpeta
        # compartida `r["hallucinated"]` sin namespacing). Al usar una subcarpeta
        # namespaced por `run_label` para no mezclar las alucinaciones de distintas
        # candidatas (ver docstring del modulo), hay que copiarlo ahi tambien -- si
        # no, Simod falla con "File .../configuration_generated.yaml does not exist"
        # (encontrado en la practica, RunningExample, 2026-08-12).
        shared_config = r["hallucinated"] / configuration_filename
        if not shared_config.exists():
            raise FileNotFoundError(
                f"No se encontro {shared_config} -- se esperaba encontrar ahi el "
                "archivo de configuracion estatico de Simod para este log (uno por "
                "log, no se genera por codigo)."
            )
        shutil.copy(shared_config, hallucinated_folder / configuration_filename)

        bps_tobe_folder = r["bps_tobe"] / run_label
        dgp.generate_bps_model(hallucinated_folder, bps_tobe_folder, configuration_filename)

        merged_name = f"{params.name}_merged_{run_label}.json"
        dgp.adapt_resources(
            original_folder=bps_asis_folder,
            original_filename=r["bpmn"],
            generated_folder=bps_tobe_folder,
            generated_filename=r["bpmn"],
            merged_filename=merged_name,
        )
        if neutralize_instantaneous_resources:
            # adapt_resources escribe el merged json dentro de
            # bps_tobe_folder/<timestamp>/best_result/ (mismo anidado que usa Simod
            # para sus propias salidas, ver dg_prediction.py::adapt_resources) -- no
            # directamente en bps_tobe_folder.
            merged_json_matches = list(bps_tobe_folder.glob(f"*/best_result/{merged_name}"))
            if not merged_json_matches:
                raise FileNotFoundError(
                    f"No se encontro {merged_name} dentro de {bps_tobe_folder}/*/best_result/ "
                    "-- confirma que adapt_resources ya corrio."
                )
            neutralize_instantaneous_task_resources(merged_json_matches[0])

        handle = ArmRunHandle(
            label=run_label,
            params=params,
            hallucinated_folder=hallucinated_folder,
            bps_folder=bps_tobe_folder,
            resources_filename=merged_name,
        )
    else:
        if existing_handle is None:
            raise ValueError("discover=False requiere existing_handle (BPS ya descubierto).")
        handle = existing_handle

    start = len(handle.stats_csv_paths) + 1
    for i in range(start, start + replicas):
        rep_output = r["simulation"] / f"{run_label}_rep{i}"
        dgp.simulate_model(
            input_path=handle.bps_folder,
            output_path=rep_output,
            bpmn_filename=r["bpmn"],
            resources_filename=handle.resources_filename,
            total_cases=params.effective_tobe_cases,
        )
        handle.stats_csv_paths.append(rep_output / f"{params.name}_prosimos_stats.csv")

    return handle


def reconstruct_tobe_handle(params: Params, run_label: str) -> ArmRunHandle:
    """
    Reconstruye un `ArmRunHandle` desde disco para un `run_label` que ya corrio
    (tipicamente en un proceso Python separado, ej. `run_screening.py`) -- permite
    que `run_final_comparison.py` reutilice el BPS ya descubierto para la candidata
    ganadora del screening sin volver a alucinar/descubrir, sin depender de que
    ambos programas corran en el mismo proceso (que es lo que hacia `orchestrate.py`
    al mantener el `ArmRunHandle` en memoria de principio a fin).

    Asume la misma convencion de nombres que `run_tobe_arm` usa al escribir:
    `bps_tobe/<run_label>`, `<name>_merged_<run_label>.json`,
    `simulation/<run_label>_rep{i}/<name>_prosimos_stats.csv`.
    """
    r = params.routes
    stats_csv_paths = sorted(
        r["simulation"].glob(f"{run_label}_rep*/{params.name}_prosimos_stats.csv"),
        key=lambda p: int(p.parent.name.rsplit("rep", 1)[1]),
    )
    return ArmRunHandle(
        label=run_label,
        params=params,
        hallucinated_folder=r["hallucinated"] / run_label,
        bps_folder=r["bps_tobe"] / run_label,
        resources_filename=f"{params.name}_merged_{run_label}.json",
        stats_csv_paths=list(stats_csv_paths),
    )
