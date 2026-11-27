# Experimentos de comparación: RULE-GAN, Dynamics y CVAE

Cada enfoque tiene su carpeta con todo lo necesario para generar logs. Lo que
comparten los tres (logs, división entrenamiento/test y reglas) está en `comun/`.

```
experimentos/
  comun/
    preparar_logs_dynamics.py     log y división comunes de los logs de Dynamics
    preparar_logs_cvae.py         log y división comunes de BPIC2012 y Traffic Fines
    preparar_validacion.py        validación (último 20% del train) en los logs de Dynamics
    <log>/log.csv                 (generado, no versionado)
    <log>/split/                  {train,subtrain,val,test}_caseids.txt (versionado)
    <log>/reglas/*.ini            reglas a probar (versionado)
  dynamics/
    descargar_paquete.py          paquete de Zenodo + modelos LSTM publicados
    predictor_corregido.py        corrección B de la regla de aceptación
    generar.py
    modelos/, ejecuciones/        (no versionados)
  rule_gan/
    entrenar.py, generar.py
    seleccion.py                  evaluación contra la validación (métricas, objetivo)
    buscar_amplia.py              Fase 1a: búsqueda amplia con TPE (corrida en RE y PE)
    buscar_grilla.py              Fase 1b: grilla de 12 configuraciones, en cada log
    busqueda/<log>/<etiqueta>/    ensayos.csv, evaluaciones.csv (versionados)
    modelos/, ejecuciones/        (no versionados; tampoco busqueda/**/modelos/)
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

## Selección del modelo RULE-GAN

Se elige **una configuración por log**, con la validación (el último 20% cronológico
del train; `comun/preparar_validacion.py`). El test no se usa para elegir.

1. **Fase 1a: búsqueda amplia** (`buscar_amplia.py`, ya corrida en RunningExample y
   PurchasingExample). TPE con 30 configuraciones distintas por log, sobre el espacio de
   V2–V6. Resultados en `rule_gan/busqueda/<log>/seleccion/`.
2. **Fase 1b: grilla reducida** (`buscar_grilla.py`, en cada log). Con los resultados de
   la fase 1a se fijaron los hiperparámetros que dieron el mismo mejor valor en ambos
   logs. Quedan 12 configuraciones y se prueban todas. Las que ya se evaluaron en la
   fase 1a se reutilizan (solo se evalúan las temperaturas que faltan).

```bash
python experimentos/rule_gan/buscar_grilla.py --log RunningExample --plan   # qué se entrena y qué se reutiliza
python experimentos/rule_gan/buscar_grilla.py --log RunningExample
```

**Hiperparámetros de la grilla:**

| Hiperparámetro | Valores | Motivo |
|---|---|---|
| Cabezas de atención | {2, 4} | La mejor depende del log (RE: 4, PE: 2) |
| ff_dim | {128, 256} | 64 fue la peor en ambos logs |
| Tasas de Adam (crítico, generador) | (1e-4, 2e-4), (1e-4, 1e-4), (4e-4, 1e-4) | Las de V2–V6; iguales, como el valor por defecto de WGAN-GP; crítico más rápido, como en SAGAN |
| d_model | 64 (fijo) | Mejor en ambos logs |
| Bloques | 3 (fijo) | Mejor en RE; empate en PE |
| time2vec_dim | 16 (fijo) | Mejor en ambos logs |
| bg_lambda, ct_lambda | 0 (fijos, sin pérdidas auxiliares) | 0 fue lo mejor en ambos logs; ct_lambda = 1, lo peor |
| Betas de Adam | (0; 0,9) (fijas) | WGAN-GP (Gulrajani et al. 2017) |
| Dropout, dimensión latente, lote, pasos del crítico, λ del gradient penalty | 0,1; 64; 32; 5; 10 (fijos) | Iguales en V2–V6 |
| Iteraciones | 1.000 (fijo), con checkpoints cada 100 | En BPIC2012 no mejoró con más iteraciones (piloto de 5.000) |

**Evaluación de cada configuración** (`seleccion.py`): se elige el checkpoint con T = 1 y
después la temperatura en {0,5; 0,6; 0,7; 0,85; 1,15; 1,3}, con 5 logs generados del
tamaño de la validación. El objetivo es el promedio de RED, CTD, 2GD, frecuencia de
actividades, 1-CONF y diferencia de diversidad, cada una dividida por la de V4 (menor es
mejor).

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
