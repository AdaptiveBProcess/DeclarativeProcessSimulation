import argparse
from pathlib import Path
from typing import Optional, List, Union

from dg_prediction import (
    generate_bps_model,
    adapt_resources,
    simulate_model,
    simulate_bimp,
    extract_rules,
    call_predict,
    compress_csv_to_gz,
    pretty_print_params,
)

import support_modules.predictor_adapter as pa

# ---------------------------------------------------------------------
# Params structure (single source of truth)
# ---------------------------------------------------------------------

from params.Params import Params

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def list_available_logs(root: Path) -> List[str]:
    logs_dir = root / "data" / "0.logs"
    if not logs_dir.exists():
        return []

    return sorted(p.name for p in logs_dir.iterdir() if p.is_dir())

# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

def cmd_list_logs(args):
    root = Path(args.root)
    logs = list_available_logs(root)

    if not logs:
        print("No logs found.")
        return

    print("Available logs:")
    for log in logs:
        print(f"  - {log}")


def cmd_run(args):
    ROOT = Path(args.root)
    params = Params(
        root=ROOT,
        log_filename=args.log,
        rep=args.rep,
        variant=args.variant,
    )

    r = params.routes
    sim = params.simulation

    rules_name = extract_rules(r["rules"])
    pretty_print_params(sim)

    # ---- prediction ----
    call_predict(
        sim,
        input_folder=r["models"],
        output_folder=r["hallucinated"],
        rules_path=r["rules"],
        root_path=params.root,
    )

    # ---- compress logs ----
    compress_csv_to_gz(r["log"] / params.log_filename, output_folder=r["input"])
    compress_csv_to_gz(r["hallucinated"] / params.log_filename)

    # ---- generate BPS models ----
    generate_bps_model(
        input_folder=r["input"],
        output_folder=r["bps_asis"],
        config_file_name="configuration_original.yaml",
    )

    generate_bps_model(
        input_folder=r["hallucinated"],
        output_folder=r["bps_tobe"],
        config_file_name="configuration_generated.yaml",
    )

    # ---- merge resources ----
    adapt_resources(
        original_folder=r["bps_asis"],
        original_filename=r["bpmn"],
        generated_folder=r["bps_tobe"],
        generated_filename=r["bpmn"],
        merged_filename=r["merged"],
    )

    # ---- simulate ----
    simulate_model(
        input_path=r["bps_tobe"],
        output_path=r["simulation"] / rules_name,
        bpmn_filename=r["bpmn"],
        resources_filename=r["merged"],
    )

    simulate_bimp(
        input_path=r["bps_tobe"],
        output_path=r["simulation"] / rules_name,
        NAME=params.name,
        PATH=params.root,
        bimp_path=args.bimp_jar,
    )

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="bps-cli",
        description="Run the BPS pipeline using a single Params structure",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---------- list-logs ----------
    p = sub.add_parser("list-logs", help="List available logs")
    p.add_argument("--root", default=".", help="Project root")
    p.set_defaults(func=cmd_list_logs)

    # ---------- run ----------
    p = sub.add_parser("run", help="Run full pipeline with defaults")
    p.add_argument("--log", default="PurchasingExample.csv", help="CSV log filename")
    p.add_argument("--root", default=".", help="Project root")
    p.add_argument("--rep", type=int, default=1, help="Simulation repetitions")
    p.add_argument("--variant", default="Rules Based Random Choice")
    p.add_argument(
        "--bimp-jar",
        default="./GenerativeLSTM/external_tools/bimp/qbp-simulator-engine.jar",
    )
    p.set_defaults(func=cmd_run)

    return parser

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
