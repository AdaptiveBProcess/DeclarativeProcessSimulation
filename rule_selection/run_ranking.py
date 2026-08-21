"""
Programa modular #b: candidatas (MINERful una sola vez) -> cumplimiento por traza ->
uplift y score de Welch -> ranking top-k (CLAUDE.md 12.2-12.4). No requiere Docker,
solo Java (para MINERful) -- es la etapa mas barata del pipeline.

Separado de `run_screening.py` (que hace este mismo paso internamente si no se le
pasa `--ranking-csv`) para poder revisar las reglas recomendadas ANTES de comprometer
tiempo de simulacion en ellas.

Ejemplo (correr desde la raiz del repo, con el env `deep_generator` activo):

    python -m rule_selection.run_ranking --log PurchasingExample.csv --s-min 0.05 --top-k 5

Salida: data/5.rule_selection/<log>/{candidates_all.csv, candidates_supported.csv, uplift_ranking.csv}.
`uplift_ranking.csv` es el archivo que despues se pasa a `run_screening.py --ranking-csv`
(programa #c) para simular esas candidatas sin recomputar este paso.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rule_selection.run_ranking")
    parser.add_argument("--root", default=".", help="Raiz del proyecto")
    parser.add_argument("--log", required=True, help="Nombre del CSV del log (ej. PurchasingExample.csv)")
    parser.add_argument(
        "--minerful-root", default=None, help="Carpeta con MINERful.jar/lib (default: <root>/MINERful)"
    )
    parser.add_argument("--s-min", type=float, default=0.05)
    parser.add_argument("--c-min", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    import pandas as pd

    from rule_selection import artifacts
    from rule_selection.ranking_pipeline import compute_ranking

    minerful_root = Path(args.minerful_root) if args.minerful_root else root / "MINERful"
    log_name = args.log.replace(".csv", "")

    print(f"Log: {args.log}")
    print(f"s_min={args.s_min}  c_min={args.c_min}  top_k={args.top_k}")
    print("\nCorriendo MINERful una sola vez + cumplimiento por traza + score de Welch...\n")

    df_candidates, ranked = compute_ranking(
        root, args.log, minerful_root, s_min=args.s_min, c_min=args.c_min, top_k=args.top_k
    )

    candidates_all_path = artifacts.rule_selection_dir(root, log_name) / "candidates_all.csv"
    n_all = len(pd.read_csv(candidates_all_path)) if candidates_all_path.exists() else None

    if n_all is not None:
        print(f"Candidatas totales descubiertas por MINERful: {n_all}")
    print(f"Candidatas soportadas por el simulador (8 plantillas): {len(df_candidates)}")
    print(
        f"\n=== Top-{len(ranked)} candidatas recomendadas "
        f"(guardado en data/5.rule_selection/{log_name}/uplift_ranking.csv) ==="
    )
    print(ranked[["candidate_id", "template", "arg0", "arg1", "score", "n_cumple", "n_no_cumple"]].to_string(index=False))


if __name__ == "__main__":
    main()
