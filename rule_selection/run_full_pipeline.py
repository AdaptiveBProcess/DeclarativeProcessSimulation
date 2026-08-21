"""
Orquestador robusto del pipeline completo GENESIS (CLAUDE.md Sec. 12): entrena LSTM ->
ranking (top-k reglas) -> screening por simulacion -> comparacion final de 4 brazos.

Reemplaza a `orchestrate.py` (corria todo en un solo proceso Python de larga duracion --
el mismo diseno que causo el crash de memoria real en BPI_Challenge_2012, CLAUDE.md
25.4-25.6: `get_stats_log_traces` relee/reconcatena archivos de trazas generadas en cada
alucinacion, y esa acumulacion crece sin limite dentro de un mismo proceso largo). Este
orquestador NO reimplementa nada -- solo secuencia llamadas a `subprocess.run(...)`
contra los programas modulares ya existentes y probados (dg_training.py, run_ranking.py,
asis_config_test.py, manual_arm_test.py, run_final_arm.py,
run_final_comparison_consolidate.py), una candidata/brazo por subproceso -- exactamente
el patron manual que ya funciono para BPI_Challenge_2012.

Por defecto salta cualquier etapa que ya tiene resultado en disco (ver las funciones
`_..._done` mas abajo) -- volver a correr el mismo comando despues de un corte retoma
donde quedo, sin reentrenar el LSTM (horas) ni resimular candidatas/brazos ya
completados. Flags `--force-*` fuerzan rehacer una etapa puntual.

Ejemplos (correr desde la raiz del repo, con el env `deep_generator` activo):

    # Corrida rigurosa completa (poblacion real, 200 epochs, 5 replicas de screening,
    # 20 de comparacion final)
    python -m rule_selection.run_full_pipeline --log RunningExample.csv

    # Corrida rapida de supervision
    python -m rule_selection.run_full_pipeline --log RunningExample.csv ^
        --hallucination-cases 30 --tobe-cases 30 --epochs 2 --max-eval 1 ^
        --screening-replicas 2 --final-replicas 2

    # Reintentar despues de un corte a mitad de camino -- mismo comando, saltea
    # automaticamente todo lo que ya esta hecho
    python -m rule_selection.run_full_pipeline --log RunningExample.csv

    # Forzar rehacer solo 2 candidatas puntuales del screening, sin tocar
    # entrenamiento/ranking/AS-IS/las otras 3 candidatas/comparacion final
    python -m rule_selection.run_full_pipeline --log RunningExample.csv ^
        --force-rescreen notchainsuccession_Task_C_Task_B,notchainsuccession_Task_C_Task_E

    # Forzar rehacer solo el brazo Placebo de la comparacion final
    python -m rule_selection.run_full_pipeline --log RunningExample.csv --force-final placebo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ARM_NAMES = ["asis", "placebo", "topsupport", "genesis"]
ARM_DISPLAY_NAMES = {
    "asis": "AS-IS",
    "placebo": "Placebo",
    "topsupport": "Top-Support",
    "genesis": "GENESIS",
}


def _run(cmd: list[str], root: Path, description: str) -> None:
    print(f"\n>>> {description}")
    print("    " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)


def _clean_arg1(value):
    return None if pd.isna(value) else value


def _parse_force_list(value: str | None, all_ids: list[str]) -> set[str]:
    if not value:
        return set()
    if value.strip().lower() == "all":
        return set(all_ids)
    return {v.strip() for v in value.split(",") if v.strip()}


# ---------------------------------------------------------------------------
# Chequeos de "ya hecho" -- puro filesystem/CSV, nada de TensorFlow/Docker aca.
# ---------------------------------------------------------------------------

def _training_done(root: Path, log_name: str) -> bool:
    """
    Una subcarpeta directa bajo data/1.predicton_models/<log_name>/ con TANTO
    <log_name>.h5 COMO parameters/model_parameters.json adentro. Ni la existencia de
    la carpeta padre ni el mtime alcanzan: ModelTrainer crea una carpeta scratch (por
    cada trial de busqueda bayesiana) y una carpeta final por separado -- si el
    entrenamiento se cae a mitad de camino, la scratch queda huerfana como hermana de
    cualquier corrida anterior ya completa, y get_latest_output_folder no distingue
    una de otra por mtime solo.
    """
    models_root = root / "data" / "1.predicton_models" / log_name
    if not models_root.exists():
        return False
    for sub in models_root.iterdir():
        if not sub.is_dir():
            continue
        if (sub / f"{log_name}.h5").exists() and (sub / "parameters" / "model_parameters.json").exists():
            return True
    return False


def _ranking_done(root: Path, log_name: str) -> bool:
    return (root / "data" / "5.rule_selection" / log_name / "uplift_ranking.csv").exists()


def _asis_done(root: Path, log_name: str, asis_label: str) -> bool:
    asis_dir = root / "data" / "3.bps_asis" / log_name / asis_label
    if not asis_dir.exists():
        return False
    return any(asis_dir.glob(f"*/best_result/{log_name}.json"))


def _stats_paths_for_run_label(root: Path, log_name: str, run_label: str) -> list[Path]:
    sim_root = root / "data" / "4.simulation_results" / log_name
    if not sim_root.exists():
        return []
    return sorted(
        sim_root.glob(f"{run_label}_rep*/{log_name}_prosimos_stats.csv"),
        key=lambda p: int(p.parent.name.rsplit("_rep", 1)[-1]),
    )


def _screening_row_done(root: Path, log_name: str, candidate_id: str, replicas: int) -> bool:
    summary_path = root / "data" / "5.rule_selection" / log_name / "screening_summary.csv"
    if not summary_path.exists():
        return False
    df = pd.read_csv(summary_path)
    match = df[df["candidate_id"] == candidate_id]
    if match.empty:
        return False
    return int(match.iloc[0]["n_replicas"]) >= replicas


def _final_arm_done(root: Path, log_name: str, display_name: str, replicas: int) -> bool:
    """
    `final_comparison_partial.csv`'s n_replicas es la unica senal confiable -- contar
    carpetas de replica a mano sobre/subestima para asis/placebo/topsupport, que
    (a diferencia de GENESIS) no limpian replicas viejas al re-descubrir.
    """
    partial_path = root / "data" / "5.rule_selection" / log_name / "final_comparison_partial.csv"
    if not partial_path.exists():
        return False
    df = pd.read_csv(partial_path)
    match = df[df["brazo"] == display_name]
    if match.empty:
        return False
    return int(match.iloc[0]["n_replicas"]) >= replicas


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

def stage_train(root: Path, log_filename: str, log_name: str, epochs: int, max_eval: int, force: bool) -> None:
    if not force and _training_done(root, log_name):
        print(f"\n[SKIP] Entrenamiento LSTM ya completo para {log_name}.")
        return
    _run(
        [sys.executable, "dg_training.py", "-f", log_filename, "-p", str(epochs), "-e", str(max_eval)],
        root,
        f"Entrenando LSTM ({epochs} epochs, {max_eval} evals bayesianos) para {log_name}...",
    )


def stage_rank(
    root: Path, log_filename: str, log_name: str, minerful_root: Path,
    s_min: float, c_min: float, top_k: int, force: bool,
) -> None:
    if not force and _ranking_done(root, log_name):
        print("\n[SKIP] Ranking ya calculado (uplift_ranking.csv existe).")
        return
    _run(
        [
            sys.executable, "-m", "rule_selection.run_ranking",
            "--root", str(root), "--log", log_filename,
            "--minerful-root", str(minerful_root),
            "--s-min", str(s_min), "--c-min", str(c_min), "--top-k", str(top_k),
        ],
        root,
        "Descubriendo candidatas (MINERful una vez) + ranking por score de Welch...",
    )


def stage_write_candidate_inis(root: Path, ranking: pd.DataFrame, log_name: str) -> None:
    """
    Ninguna etapa anterior escribe los .ini de las candidatas -- compute_ranking()
    solo calcula el ranking; write_rule_ini() hoy solo se invoca desde
    simulation_arms.make_candidate_params(), que manual_arm_test.py (usado por
    stage_screening) no pasa por -- espera el .ini ya existente en
    data/0.logs/<log>/<candidate_id>.ini. Escribirlos aca es barato (texto plano, sin
    TensorFlow/Docker), no necesita subproceso propio.
    """
    from rule_selection.candidates import write_rule_ini

    rules_dir = root / "data" / "0.logs" / log_name
    for _, row in ranking.iterrows():
        out_path = rules_dir / f"{row['candidate_id']}.ini"
        write_rule_ini(row["template"], row["arg0"], _clean_arg1(row.get("arg1")), out_path)


def stage_asis(
    root: Path, log_filename: str, log_name: str, asis_config_filename: str,
    asis_run_label: str, tobe_cases: int | None, force: bool,
) -> None:
    if not force and _asis_done(root, log_name, asis_run_label):
        print(f"\n[SKIP] AS-IS ya descubierto (run_label={asis_run_label}).")
        return
    cmd = [
        sys.executable, "-m", "rule_selection.asis_config_test",
        "--root", str(root), "--log", log_filename,
        "--config-filename", asis_config_filename,
        "--run-label", asis_run_label, "--replicas", "1",
    ]
    if tobe_cases is not None:
        cmd += ["--tobe-cases", str(tobe_cases)]
    _run(cmd, root, f"Descubriendo AS-IS con Simod (run_label={asis_run_label})...")


def stage_screening(
    root: Path, log_filename: str, log_name: str, ranking: pd.DataFrame,
    screening_replicas: int, hallucination_cases: int | None, tobe_cases: int | None,
    asis_run_label: str, tobe_config_filename: str, force_ids: set[str],
) -> tuple[list[str], list[str]]:
    from rule_selection import artifacts
    from rule_selection.kpi import mean_pce_over_replicas

    succeeded: list[str] = []
    failed: list[str] = []
    total = len(ranking)

    for position, (_, candidate) in enumerate(ranking.iterrows(), start=1):
        candidate_id = candidate["candidate_id"]
        run_label = f"screening_{candidate_id}"
        force = "all" in force_ids or candidate_id in force_ids

        if not force and _screening_row_done(root, log_name, candidate_id, screening_replicas):
            print(f"\n[SKIP] Candidata {position}/{total} ({candidate_id}) ya screeneada.")
            succeeded.append(candidate_id)
            continue

        # Reparacion barata: si el filesystem ya tiene suficientes replicas (ej. el
        # orquestador se corto justo despues de que el subproceso terminara Prosimos,
        # antes de reconstruir esta fila), no relanzar el subproceso completo -- solo
        # recalcular el PCE medio a partir de lo que ya esta en disco.
        existing_stats = _stats_paths_for_run_label(root, log_name, run_label)
        if not force and len(existing_stats) >= screening_replicas:
            print(
                f"\n[REPARAR] Candidata {position}/{total} ({candidate_id}): ya hay "
                f"{len(existing_stats)} replicas en disco, reconstruyendo su fila sin relanzar."
            )
            mean_pce = mean_pce_over_replicas(existing_stats[:screening_replicas])
            artifacts.upsert_screening_summary_row(root, log_name, {
                "candidate_id": candidate_id,
                "template": candidate["template"],
                "arg0": candidate["arg0"],
                "arg1": _clean_arg1(candidate.get("arg1")),
                "score": candidate.get("score"),
                "screening_mean_pce": mean_pce,
                "n_replicas": screening_replicas,
            })
            succeeded.append(candidate_id)
            continue

        cmd = [
            sys.executable, "-m", "rule_selection.manual_arm_test",
            "--root", str(root), "--log", log_filename,
            "--rule", f"{candidate_id}.ini", "--run-label", run_label,
            "--replicas", str(screening_replicas),
            "--asis-run-label", asis_run_label,
            "--tobe-config-filename", tobe_config_filename,
        ]
        if hallucination_cases is not None:
            cmd += ["--hallucination-cases", str(hallucination_cases)]
        if tobe_cases is not None:
            cmd += ["--tobe-cases", str(tobe_cases)]

        try:
            _run(
                cmd, root,
                f"Screening {position}/{total}: {candidate_id} ({screening_replicas} replicas)...",
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"\n[FALLO] Candidata {candidate_id} fallo (exit code {exc.returncode}) -- "
                "se continua con las restantes (screening tolera fallas individuales)."
            )
            failed.append(candidate_id)
            continue

        stats_paths = _stats_paths_for_run_label(root, log_name, run_label)
        mean_pce = mean_pce_over_replicas(stats_paths)
        artifacts.upsert_screening_summary_row(root, log_name, {
            "candidate_id": candidate_id,
            "template": candidate["template"],
            "arg0": candidate["arg0"],
            "arg1": _clean_arg1(candidate.get("arg1")),
            "score": candidate.get("score"),
            "screening_mean_pce": mean_pce,
            "n_replicas": len(stats_paths),
        })
        succeeded.append(candidate_id)

    return succeeded, failed


def stage_final_arm(
    root: Path, log_filename: str, log_name: str, arm: str, final_replicas: int,
    hallucination_cases: int | None, tobe_cases: int | None,
    asis_run_label: str, tobe_config_filename: str, force: bool,
) -> None:
    display_name = ARM_DISPLAY_NAMES[arm]
    if not force and _final_arm_done(root, log_name, display_name, final_replicas):
        print(f"\n[SKIP] Brazo {display_name} ya tiene {final_replicas}+ replicas registradas.")
        return

    cmd = [
        sys.executable, "-m", "rule_selection.run_final_arm",
        "--root", str(root), "--log", log_filename, "--arm", arm,
        "--replicas", str(final_replicas),
        "--tobe-config-filename", tobe_config_filename,
    ]
    if arm != "genesis":
        cmd += ["--asis-run-label", asis_run_label]
    if hallucination_cases is not None:
        cmd += ["--hallucination-cases", str(hallucination_cases)]
    if tobe_cases is not None:
        cmd += ["--tobe-cases", str(tobe_cases)]

    _run(cmd, root, f"Brazo final: {display_name} ({final_replicas} replicas)...")


def stage_consolidate(root: Path, log_filename: str, log_name: str, force: bool) -> None:
    out_path = root / "data" / "5.rule_selection" / log_name / "final_comparison.csv"
    if not force and out_path.exists():
        print("\n[SKIP] final_comparison.csv ya existe.")
        return
    _run(
        [
            sys.executable, "-m", "rule_selection.run_final_comparison_consolidate",
            "--root", str(root), "--log", log_filename,
        ],
        root, "Consolidando comparacion final (4 brazos, deltas causales)...",
    )


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.run_full_pipeline")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. RunningExample.csv)")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument("--minerful-root", default=None, help="Default: <root>/MINERful")
    parser.add_argument("--s-min", type=float, default=0.05)
    parser.add_argument("--c-min", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--screening-replicas", type=int, default=5)
    parser.add_argument("--final-replicas", type=int, default=20)
    parser.add_argument(
        "--hallucination-cases", type=int, default=None,
        help="Trazas a alucinar por candidata/brazo (default: poblacion real del log)",
    )
    parser.add_argument(
        "--tobe-cases", type=int, default=None,
        help="Casos a simular en Prosimos, AS-IS y TO-BE por igual (default: poblacion real del log)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Epochs de entrenamiento LSTM (dg_training.py -p)")
    parser.add_argument("--max-eval", type=int, default=10, help="Evaluaciones de busqueda bayesiana (dg_training.py -e)")
    parser.add_argument("--asis-config-filename", default="configuration_original.yaml")
    parser.add_argument("--tobe-config-filename", default="configuration_generated.yaml")
    parser.add_argument(
        "--asis-run-label", default=None,
        help="Default: pipeline_<log>_asis. Pasar uno ya existente para reutilizar un AS-IS de otra corrida.",
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-rerank", action="store_true")
    parser.add_argument("--force-asis", action="store_true")
    parser.add_argument(
        "--force-rescreen", default=None,
        help="'all' o candidate_id separados por coma -- fuerza rehacer esas candidatas del screening",
    )
    parser.add_argument(
        "--force-final", default=None,
        help="'all' o brazos separados por coma (asis,placebo,topsupport,genesis)",
    )
    parser.add_argument("--force-consolidate", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    minerful_root = Path(args.minerful_root) if args.minerful_root else root / "MINERful"
    log_name = args.log.replace(".csv", "")
    asis_run_label = args.asis_run_label or f"pipeline_{log_name}_asis"

    print(f"=== Pipeline completo GENESIS: {args.log} ===")
    print(f"root={root}")
    print(f"asis_run_label={asis_run_label}")

    stage_train(root, args.log, log_name, args.epochs, args.max_eval, args.force_retrain)

    stage_rank(root, args.log, log_name, minerful_root, args.s_min, args.c_min, args.top_k, args.force_rerank)
    ranking = pd.read_csv(root / "data" / "5.rule_selection" / log_name / "uplift_ranking.csv")

    stage_write_candidate_inis(root, ranking, log_name)

    stage_asis(root, args.log, log_name, args.asis_config_filename, asis_run_label, args.tobe_cases, args.force_asis)

    force_screen_ids = _parse_force_list(args.force_rescreen, list(ranking["candidate_id"]))
    succeeded, failed = stage_screening(
        root, args.log, log_name, ranking, args.screening_replicas,
        args.hallucination_cases, args.tobe_cases, asis_run_label,
        args.tobe_config_filename, force_screen_ids,
    )
    if failed:
        print(f"\n[ADVERTENCIA] {len(failed)} candidata(s) fallaron en screening: {failed}")
    if not succeeded:
        print("\n[DETENIDO] Ninguna candidata de screening tuvo exito -- no se puede elegir GENESIS.")
        sys.exit(1)

    force_final_arms = _parse_force_list(args.force_final, ARM_NAMES)
    for arm in ARM_NAMES:
        stage_final_arm(
            root, args.log, log_name, arm, args.final_replicas,
            args.hallucination_cases, args.tobe_cases, asis_run_label,
            args.tobe_config_filename,
            force=("all" in force_final_arms or arm in force_final_arms),
        )

    stage_consolidate(root, args.log, log_name, args.force_consolidate)

    final_path = root / "data" / "5.rule_selection" / log_name / "final_comparison.csv"
    print("\n=== COMPLETO ===")
    print(pd.read_csv(final_path).to_string(index=False))


if __name__ == "__main__":
    main()
