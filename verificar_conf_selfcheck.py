# -*- coding: utf-8 -*-
"""
Self-check de CONF: compara test_split.csv contra si mismo.

Por que esta prueba es diagnostica
-----------------------------------
CONF mina reglas DECLARE con MINERful exigiendo soporte >= 90% sobre el log
de referencia (test_split.csv), y luego verifica cuantas trazas del log
simulado cumplen esas reglas. Si el log "simulado" es el MISMO test_split.csv,
el resultado deberia acercarse a ~90%+ casi por construccion (las reglas se
descubrieron exigiendo que al menos el 90% de esas mismas trazas las cumplan).

Si este self-check da un CONF bajo, el problema esta en el calculo de la
metrica (evaluacion/metrics.py), no en el generador GAN. Si da cercano a
90%+, el calculo esta sano y el CONF reportado para el log del GAN (48.7%)
es un resultado real del generador que hay que investigar por otro lado.

Ejecutar desde la raiz del proyecto, en el venv deep_generator:
    python verificar_conf_selfcheck.py
"""
import sys
from pathlib import Path

TEST_SPLIT = Path("data/4.simulation_results/RunningExample/metricas/test_split.csv")


def main():
    if not TEST_SPLIT.exists():
        print(f"[ERROR] No se encontro: {TEST_SPLIT}")
        print("        Verifica que ya hayas corrido "
              "'python dg_training.py -m transformer_wgan' "
              "(ese paso genera test_split.csv).")
        sys.exit(1)

    from evaluacion.evaluator import evaluate

    print(f"Comparando {TEST_SPLIT} contra si mismo (self-check de CONF)...\n")
    resultado = evaluate(
        ref_path=str(TEST_SPLIT),
        sim_path=str(TEST_SPLIT),
        support_threshold=0.90,
        verbose=True,
    )

    conf = resultado.get("CONF")
    n_rules = resultado.get("n_rules")

    print("\n" + "=" * 70)
    if conf is None:
        print("[FAIL] CONF no se pudo calcular -- revisa la excepcion/warning "
              "impresa arriba (posible fallo de MINERful o pm4py).")
    elif conf >= 0.85:
        print(f"[OK] CONF self-check = {conf:.2%}  ({n_rules} reglas)")
        print("     Cercano al soporte minimo (90%) -- el calculo de CONF "
              "parece sano.")
        print("     El 48.7% obtenido con el log del GAN es entonces un "
              "resultado real del generador, no un artefacto de la metrica.")
    else:
        print(f"[SOSPECHOSO] CONF self-check = {conf:.2%}  ({n_rules} reglas)")
        print("     Deberia acercarse a ~90% por construccion (las reglas se "
              "minaron exigiendo soporte>=90% sobre este mismo log).")
        print("     Esto sugiere un problema en el calculo de CONF "
              "(evaluacion/metrics.py), no necesariamente en el generador.")
    print("=" * 70)


if __name__ == "__main__":
    main()
