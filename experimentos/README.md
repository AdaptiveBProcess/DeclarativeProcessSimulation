# Experimentos de comparación: RULE-GAN, Dynamics y CVAE

Cada enfoque tiene su carpeta con todo lo necesario para generar logs. Lo que
comparten los tres (logs, división entrenamiento/test y reglas) está en `comun/`.

```
experimentos/
  comun/
    preparar_logs_dynamics.py     log y división comunes de los logs de Dynamics
    <log>/log.csv                 (generado, no versionado)
    <log>/split/                  train_caseids.txt, test_caseids.txt (versionado)
    <log>/reglas/*.ini            reglas a probar (versionado)
  dynamics/
    descargar_paquete.py          paquete de Zenodo + modelos LSTM publicados
    predictor_corregido.py        corrección B de la regla de aceptación
    generar.py
    modelos/, ejecuciones/        (no versionados)
  rule_gan/
    entrenar.py, generar.py
    modelos/, ejecuciones/        (no versionados)
  cvae/
    README.md                     cómo obtener el código, la versión usada y el entorno
    requirements_windows.txt
    generar.py
    repo/, repo_icpm2024/         clon de CVAE y copia en la versión del paper (no versionados)
    ejecuciones/                  (no versionado)
```

Las ejecuciones quedan en `<enfoque>/ejecuciones/<log>/<regla>/corrida_XX/`, con el
log generado (`<log>.csv`), la regla usada (`regla.ini`) y `meta.json` (semilla,
modelo, tiempo, número de trazas).

## Entornos

| Enfoque | Entorno conda | Notas |
|---|---|---|
| Dynamics, RULE-GAN | `deep_generator` (Python 3.9, TensorFlow 2.13) | Requiere el submódulo `GenerativeLSTM` inicializado |
| CVAE | `cvae` (Python 3.11, PyTorch 2.0.1) | Ver `cvae/README.md` |

## Pasos (RunningExample)

```bash
# 1. Dynamics: paquete de Zenodo, modelos publicados y log/división comunes
python experimentos/dynamics/descargar_paquete.py
python experimentos/comun/preparar_logs_dynamics.py

# 2. Dynamics: generar (sin reentrenar)
python experimentos/dynamics/generar.py --log RunningExample --regla r1_directly_TaskB_TaskC --corridas 3

# 3. RULE-GAN: entrenar una vez por log y generar por regla
python experimentos/rule_gan/entrenar.py --log RunningExample
python experimentos/rule_gan/generar.py --log RunningExample --regla r1_directly_TaskB_TaskC --corridas 3

# 4. CVAE (entorno cvae): generar con sus modelos publicados (ver cvae/README.md)
python experimentos/cvae/generar.py --dataset bpic2012_a --corridas 3
```

## Decisiones

- **Log y división comunes.** En los logs de Dynamics se usa exactamente su log
  (sin los eventos artificiales de inicio y fin) y su test publicado: en
  RunningExample, los últimos 107 casos de 540. RULE-GAN y CVAE se entrenan con los
  433 restantes.
- **Reglas.** Las de Dynamics, con su misma sintaxis y `variation = =1` (todas las
  trazas generadas deben cumplir la regla).
- **Número de trazas.** Cada corrida genera tantas trazas como casos de test.
- **Corrección B en Dynamics.** El generador publicado acepta una sola traza por
  proceso cuando se pide el 100% de cumplimiento (8 trazas en una CPU de 8 núcleos;
  su propio paquete muestra 3 o 4). `predictor_corregido.py` corrige solo la regla
  de aceptación; modelo, muestreo y filtro de reglas son los originales. Con
  `--publicado` se corre sin la corrección.
- **Presupuesto de intentos.** 10 veces el número de trazas pedidas, igual en
  Dynamics (corregido) y en RULE-GAN.
- **Semillas.** Cada corrida fija `random`, `numpy` y `tensorflow` con
  `semilla + corrida - 1`.
