# CLOUD.md — Bitácora de sesión (rama `gan-module`)

> Contexto persistente para Claude Code sobre el trabajo hecho en esta rama.
> Leer completo al inicio de cada sesión que retome este tema.
> Última actualización: 2026-08-03.
> **LEER PRIMERO §11** — invalida partes de §4/§5/§10: apareció el código real
> de V2-V6 (nunca comiteado, vivía sin commitear en una segunda carpeta local del
> usuario) y cambia el diagnóstico de degradación de métricas.

---

## 1. Qué pasó en esta sesión (resumen ejecutivo)

1. Se recuperó el acceso a la rama `gan-module` (no estaba perdida, solo faltaba
   crearla localmente desde `origin/gan-module`).
2. De paso, se resolvió una fusión (merge) que había quedado a medias en la rama
   `recommended_module`.
3. Se restauraron 4 archivos de `GenerativeGAN/` que habían desaparecido del disco
   (probable interferencia de sincronización de OneDrive).
4. Se hizo un análisis a fondo del pipeline GAN (`dg_training.py` → `dg_prediction.py`)
   contra el código real de la rama, cruzado con una tabla de hiperparámetros de 6
   versiones (V1-V6) que el usuario aportó. **Hallazgo crítico (histórico, ver §10
   para el estado actual)**: el código en el repo
   solo implementaba la arquitectura V1; las versiones V2-V6 (incluida V4, la
   seleccionada) nunca se comitearon — solo existen como configuración documentada.
5. **(2026-08-02)** Se reconstruyó la arquitectura V4 en código (Transformer +
   Time2Vec + WGAN-GP), reemplazando V1. Una revisión de código independiente
   encontró y se corrigió un bug preexistente en el cálculo de CONF (orden de
   argumentos Activation/Target en `evaluacion/metrics.py`). Ver §10 para el detalle
   completo — es la sección más reciente y la que hay que leer primero si se retoma
   este trabajo.

---

## 2. Recuperación de git (contexto, no repetir si ya está resuelto)

- **Rama `gan-module`**: existe en `origin/gan-module` con todo el historial de
  experimentos (`Creacion ambiente GAN`, `Implementar metrica CONF con MINERful +
  pm4py y corregir RED/2GD/CTD`, `desarrollo implementacion`). Se creó localmente con
  `git checkout -b gan-module origin/gan-module` y quedó como rama activa.
- **Merge resuelto en `recommended_module`** (commit `ef5686d`): conflicto entre un
  commit local (`bc05ec5 "ajustes"`) y 3 commits del origin (`2d5ea73`, `d7c7d81`,
  `c7864fa`). Decisiones tomadas:
  - `dg_prediction.py`: se adoptó la versión del origin — vectorización con
    `MultiIndex.isin` en vez de `.apply` fila-por-fila, y **simulación de 4 corridas**
    (ASIS + 3 réplicas TOBE) en vez de una sola corrida.
  - `MINERful/modelo_declarativo_enriquecedio_v2.py`: se adoptó la versión del origin
    (envuelta en `main()`, con `generar_modelo_global()`, y con la selección de mejor
    regla corregida a `Support+Coverage+Confidence` en vez del bug de ordenar
    alfabéticamente). Se mantuvo `RunningExample` como dataset por defecto (preferencia
    del usuario sobre `BPI_Challenge_2012`).
  - Archivos de datos regenerados (`.xes`, `.csv` de MINERful): se resolvieron a la
    versión local, consistente con `RunningExample` como dataset activo.
- **Archivos borrados restaurados**: `GenerativeGAN/__init__.py`,
  `GenerativeGAN/model_prediction/__init__.py`,
  `GenerativeGAN/model_prediction/gan_predictor.py`,
  `GenerativeGAN/model_training/__init__.py` — habían desaparecido físicamente del
  disco (la carpeta del repo está dentro de OneDrive, sospecha de interferencia de
  sincronización). Se restauraron con `git restore` desde el último commit
  (`7e16b3a "desarrollo implementacion"`), sin pérdida de contenido.
- **Nota para el futuro**: si vuelven a aparecer archivos "deleted" en `git status`
  sin que nadie los haya borrado a propósito, sospechar primero de OneDrive antes que
  de una operación de git. Considerar excluir `.git/` de la sincronización de OneDrive.
- Queda pendiente, sin tocar, un conjunto de cambios sueltos **dentro del submódulo
  `GenerativeLSTM`** (en detached HEAD): `dg_predictiction.py`, `dg_training.py`,
  `model_predictor.py` y algunos `.bpmn`/`.csv` de test modificados sin commit. No es
  parte de esta rama ni de este análisis — decidir aparte si se quiere commitear.

---

## 3. Pipeline GAN — cómo funciona realmente (verificado en código)

```
python dg_training.py -m simple_gan
    → GANTrainer (GenerativeGAN/model_training/gan_trainer.py)
    → split cronológico 70/10/20 (train+val para entrenar, test como referencia)
    → entrena generador + discriminador
    → guarda data/1.predicton_models/<NAME>/<run>/  con:
        <log>.h5                       (generador entrenado)
        parameters/model_parameters.json   (model_type: "simple_gan", index_ac, index_rl, scale_args...)
        parameters/test_log.csv
        data/4.simulation_results/<NAME>/metricas/test_split.csv

python dg_prediction.py
    → lee model_parameters.json del último modelo entrenado
    → detecta model_type == "simple_gan" → despacha a _run_gan_pipeline()
    → GANPredictor genera trazas desde ruido, sobre-muestreando (num_cases * 5)
      hasta alcanzar la proporción objetivo de cumplimiento de la regla DECLARE
      (evaluate_condition contra rules.ini)
    → evalúa el log generado contra test_split.csv con evaluacion/evaluator.py
      (RED, CTD, 2GD, CONF) → guarda metricas/metrics_<run_id>.csv
```

Archivos clave:
| Archivo | Rol |
|---|---|
| `dg_training.py::main()` | Entry point de entrenamiento; dispatcher lstm vs gan por `-m` |
| `GenerativeGAN/model_training/gan_trainer.py::GANTrainer` | Split 70/10/20, entrenamiento, export de parámetros |
| `GenerativeGAN/model_training/models/model_simple_gan.py` | Arquitectura generador/discriminador (**V4 desde 2026-08-02** — Transformer+Time2Vec, ver §10) |
| `GenerativeGAN/model_training/samples_creator.py::GANSamplesCreator` | Convierte log a tensor one-hot `(n_traces, max_len, n_ac+n_rl+2)` |
| `dg_prediction.py::main()` | Dispatcher lstm vs gan por `model_type` leído del json |
| `dg_prediction.py::call_predict_gan / _run_gan_pipeline` | Orquesta generación + evaluación GAN |
| `GenerativeGAN/model_prediction/gan_predictor.py::GANPredictor` | Genera trazas, decodifica, filtra por regla DECLARE |
| `evaluacion/metrics.py` | RED, CTD, 2GD, CONF — basado en Graziosi et al. (2024) arXiv:2411.02131 |
| `evaluacion/evaluator.py` | Orquesta `evaluate()` / `evaluate_batch()` sobre las 4 métricas |

**Confirmado**: las 4 métricas (RED, CTD, 2GD, CONF) están implementadas exactamente
como se documentan (EMD/Wasserstein-1 para RED y CTD, TVD para 2GD, conformidad
DECLARE vía MINERful + `pm4py.algo.conformance.declare` para CONF). Esta parte del
pipeline es sólida y confiable.

---

## 4. HALLAZGO CRÍTICO (HISTÓRICO, RESUELTO 2026-08-02 — ver §10) — el código solo implementaba V1, no V4 (la seleccionada)

Comparando el código real de `model_simple_gan.py` contra la tabla de hiperparámetros
del usuario (`matriz-asignaciones(Iteraciones-RULEGAN).csv`, no versionada en el repo,
ver §5 para los valores), coincide **exactamente** con la columna **V1**:
GRU(128)→GRU(64), one-hot (sin embeddings), salida de tiempo `Dense(2, sigmoid)`,
discriminador con salida `Dense(1, sigmoid)`, optimizador `Adam(lr=0.0002, β1=0.5)`,
loss `binary_crossentropy`.

Verificación de que V2-V6 nunca se comitearon:
- `git log --follow -- GenerativeGAN/model_training/models/model_simple_gan.py` →
  un único commit: `8816197 "Creacion ambiente GAN"`. Nunca se modificó después.
- Búsqueda de `WGAN`, `Wasserstein` (como arquitectura), `Time2Vec`,
  `gradient_penalty` en **todos los commits de todas las ramas** (`main`,
  `recommended_module`, `gan-module`, `legacy`, `submodule_creation`): cero
  coincidencias de implementación de modelo. Las únicas coincidencias de
  "Wasserstein" son en `evaluacion/metrics.py` (la métrica EMD de evaluación, no el
  entrenamiento).
- `GANTrainer.__init__` no lee ni conoce `d_model`, `num_heads`, `n_critic`,
  `bg_lambda`, `ct_lambda` — no existe ninguna ruta de código para invocar esas
  variantes aunque se intente vía parámetros.

**Confirmado con el usuario (2026-08-01)**: las versiones V1-V6 nunca se guardaron
como commits separados; se fueron probando localmente y al final se seleccionó **V4**
como la versión final, pero ese código no quedó persistido en git. Lo que sí quedó es
la tabla de configuración completa (ver §5), que es detallada a nivel de capa —
funciona como especificación reproducible aunque el código no exista.

**Lo que falta reconstruir no son hiperparámetros — es arquitectura y bucle de
entrenamiento completos**:

| Componente | V1 (en el repo hoy) | V4 (la seleccionada, por reconstruir) |
|---|---|---|
| Entrenamiento | GAN clásico, `binary_crossentropy`, `train_on_batch` directo | WGAN-GP: pérdida Wasserstein, 5 pasos D por 1 paso G, gradient penalty (`GradientTape` anidados), `@tf.function` |
| Generador/Discriminador | GRU apiladas | Bloques Transformer (Multi-Head Self-Attention, Pre-LN) |
| Representación actividad/rol | One-hot | Proyección densa aprendida (`TimeDistributed(Dense(32, no bias))`) |
| Representación de tiempo | `Dense(2, sigmoid)` escalar | Time2Vec(dim=16) |

**Estado**: reconstrucción de V4 **no implementada aún** — decisión explícita del
usuario de cerrar primero el análisis documental antes de tocar código (2026-08-01).
Cuando se retome, seguir la especificación de §5 (columna V4) como fuente de verdad.

---

## 5. Tabla de hiperparámetros V1-V6 (registro experimental del usuario)

> Fuente: `matriz-asignaciones(Iteraciones-RULEGAN).csv`, aportado por el usuario en
> esta sesión. No existe como archivo en el repo — se transcribe aquí para no perder
> el contexto entre sesiones.

### Resultados (métricas finales por versión)

| Métrica | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|
| RED (↓ mejor) | 0.2069 | 0.0712 | 0.0250 | **0.0202** | 0.0323 | 0.0806 |
| CTD segundos (↓ mejor) | 26,073 | 11,621 | 3,107 | **2,093** | 3,976 | 3,412 |
| 2GD (↓ mejor) | 0.8223 | 0.2777 | 0.1512 | 0.1273 | 0.1386 | **0.1169** |
| CONF (↑ mejor) | 42.82% | 71.36% | 83.36% | 87.90% | 85.59% | **90.18%** |
| n_rules (MINERful, soporte≥90%) | 41 | 41 | 41 | 41 | 41 | 41 |

### Arquitectura e hiperparámetros por versión

- **V1**: GAN clásico (no WGAN). GRU. One-hot. Tiempo escalar normalizado.
  `epochs=200, batch=32, latent_dim=100, norm=max, Adam(lr=0.0002, β1=0.5), loss=BCE`.
- **V2**: WGAN-GP (λ=10). Transformer 2 bloques, `d_model=32, heads=2, ff=64`.
  Proyección densa 16 (actividad/rol). Time2Vec(dim=8).
  `latent_dim=64, epochs=500, n_critic=5, batch=32, dropout=0.1,
  Adam D(lr=1e-4,β1=0,β2=0.9), Adam G(lr=2e-4,β1=0,β2=0.9)`.
- **V3**: igual que V2 pero `epochs=1000, num_blocks=3, ff=128, time2vec_dim=16`.
- **V4**: igual que V3 pero `d_model=64, num_heads=4, ff=256`, proyección densa 32.
  **Sin pérdidas auxiliares.**
- **V5**: igual que V4 + pérdidas auxiliares: `bg_lambda=5.0` (MSE bigrama
  generado vs. train), `ct_lambda=1.0` (`|CT_gen − CT_train|`).
  `L_G = L_adv + 5.0·MSE(bigrama_gen, bigrama_train) + 1.0·|CT_gen − CT_train|`.
- **V6**: igual que V5 pero con pesos auxiliares reducidos ~100x/10x:
  `bg_lambda=0.05, ct_lambda=0.1`.
  `L_G = L_adv + 0.05·MSE(bigrama_gen, bigrama_train) + 0.10·|CT_gen − CT_train|`.

Arquitectura generador V4-V6 (idéntica salvo pérdida auxiliar):
```
z(64) → Dense(seq×64, relu) → Reshape(seq,64) → +Pos.Encoding
   → TransformerBlock×3 [d=64, h=4, ff=256]
   → Dense(n_ac, softmax) | Dense(n_rl, softmax) | Dense(2, sigmoid)
   → Concat → (seq, n_ac + n_rl + 2)
```
Discriminador V4-V6:
```
traza(seq, n_ac+n_rl+2)
  → [:n_cat] TimeDistributed Dense(32)   → cat_proj(seq,32)
  → [-2]     Time2Vec(16)                → dur_enc(seq,16)
  → [-1]     Time2Vec(16)                → wait_enc(seq,16)
  → Concat → Dense(64) → +Pos.Encoding
  → TransformerBlock×3 [d=64, h=4, ff=256]
  → GlobalAvgPool → Dense(64, relu) → Dense(1)  [score Wasserstein]
```

Entrenamiento (V2-V6): split cronológico 70/10/20; 5 pasos D por cada paso G;
gradient penalty con `GradientTape` anidados; `@tf.function` para grafo compilado;
métricas evaluadas entre log generado y split de test.

Entrenamiento/características V1: train 378 (70%), val 54 (10%, no usado), test 108
(20%); proporción de la regla en train: 77.25% (292/378).

---

## 6. Análisis de tendencias (hallazgo propio, a partir de la tabla de §5)

- **V1→V2 es el salto más grande**: cambiar GAN clásico/GRU por WGAN-GP/Transformer/
  Time2Vec mejora las 4 métricas 55-66% de una sola vez. Es la decisión arquitectónica
  que más pesa de toda la evolución.
- **V2→V4**: aumentar capacidad (más bloques, más cabezas de atención, `d_model`
  mayor) y más epochs (500→1000) sigue mejorando las 4 métricas monótonamente.
  **V4 es el óptimo en RED (0.0202) y CTD (2,093s)**.
- **V4→V5** (pérdidas auxiliares con pesos altos `bg_lambda=5.0, ct_lambda=1.0`):
  **empeora las 4 métricas simultáneamente** respecto a V4. Evidencia de que forzar
  el ajuste de bigramas/tiempo de ciclo compite con la fidelidad de posición relativa
  de eventos (RED) y de tiempo de ciclo (CTD) — hay una tensión real entre
  conformidad de flujo de control y fidelidad temporal fina.
- **V5→V6** (mismos pesos reducidos 100x/10x): recupera CONF (90.18%, el mejor de
  toda la tabla) y 2GD (0.1169, el mejor), pero RED y CTD siguen peor que V4.
- **Conclusión**: elegir V4 como versión final es coherente con los datos — es el
  punto de mejor fidelidad temporal (RED/CTD) sin la penalización que introducen las
  pérdidas auxiliares. V6 sería la alternativa si el criterio prioritario fuera
  conformidad DECLARE (CONF) por encima de fidelidad temporal.

---

## 7. Gap metodológico abierto — "trazas más diversas" / comportamiento no visto

Ninguna de las 4 métricas (RED/CTD/2GD/CONF) mide diversidad o novedad — miden
**fidelidad distribucional** al log de referencia. Un generador que memoriza
perfectamente el log de entrenamiento puntuaría excelente en las 4 sin generar una
sola traza nueva.

Verificado en `GenerativeGAN/model_prediction/gan_predictor.py`: no existe ningún
filtro que compare una traza generada contra las variantes ya presentes en el log
original para confirmar que representa comportamiento no visto. La propiedad es
plausible por diseño (se genera desde ruido, no por replay), pero **no está medida
ni verificada en ningún punto del pipeline actual**.

**Pendiente**: diseñar una métrica de novedad/diversidad si se quiere sostener esa
afirmación con evidencia (ver opción descartada en AskUserQuestion de esta sesión —
el usuario prefirió cerrar el análisis antes de abordarlo).

---

## 8. Próximos pasos

1. [x] **Reconstruir V4 en código** — hecho 2026-08-02, ver §10.
2. **Diseñar métrica de novedad/diversidad** (§7) si se va a defender la afirmación
   de "comportamiento no visto" con evidencia, no solo por diseño arquitectónico.
   Sigue pendiente.
3. Decidir qué hacer con los cambios sueltos del submódulo `GenerativeLSTM` (§2,
   detached HEAD, sin commit) — no bloquea lo anterior pero sigue pendiente.
4. **(nuevo, 2026-08-02)** Ejecutar `dg_training.py -m simple_gan` +
   `dg_prediction.py` con `RunningExample.csv` en el venv `deep_generator` — ver §10
   para los comandos exactos y qué revisar en los resultados antes de dar la corrida
   por buena.
5. **(nuevo, 2026-08-02)** Verificar con una prueba sintética de 15 minutos que el
   fix de Activation/Target en `evaluacion/metrics.py` (§10) da el resultado correcto
   para al menos una regla de la familia Precedence y una de Response/Succession con
   una traza de ejemplo construida a mano — el fix se razonó y se verificó contra el
   código fuente de pm4py y contra datos reales del repo, pero no se pudo *ejecutar*
   (pm4py no está instalado en el entorno de herramientas de Claude Code).

---

## 9. Convenciones para Claude Code en esta rama

- Este archivo (`comentarios-cloud/CLOUD.md`) es específico de la rama `gan-module` /
  el trabajo del pipeline GAN. Es distinto de `CLAUDE.md` en la raíz del repo, que
  documenta el Paper 1 (GENESIS, política declarativa) — no mezclar contexto de
  ambos temas aunque compartan el mismo repositorio físico.
- No inventar valores de hiperparámetros de V2-V6: usar únicamente los de §5. Si se
  necesita un valor no listado ahí, marcarlo `POR CONFIRMAR` y preguntar.
- Al reconstruir V4, verificar contra la tabla de §5 capa por capa antes de dar por
  válida la implementación — es la única fuente de verdad disponible dado que el
  código original no existe.
- Mantener este archivo actualizado: al completar un punto de §8, marcarlo hecho y
  anotar el hallazgo, igual que se hace en `CLAUDE.md`.

---

## 10. Etapa 1 completada — reconstrucción de V4 y verificación (sesión 2026-08-02)

### 10.1 Qué se implementó

Se reemplazó la arquitectura V1 (GRU + one-hot + BCE) por V4 (Transformer +
Time2Vec + WGAN-GP), siguiendo exactamente la especificación de §5. Archivos
modificados:

| Archivo | Cambio |
|---|---|
| `GenerativeGAN/model_training/models/model_simple_gan.py` | Reescritura completa: capas `Time2Vec`, `PositionalEncoding`, `TransformerBlock` (Pre-LN); `build_generator`/`build_discriminator` con `d_model=64, num_heads=4, ff_dim=256, num_blocks=3`; discriminador termina en `Dense(1)` sin activación (score Wasserstein, no sigmoid). Exporta `CUSTOM_OBJECTS` para serialización. |
| `GenerativeGAN/model_training/gan_trainer.py` | `__init__` lee los nuevos hiperparámetros (defaults V4). `_train_gan` reescrito como bucle WGAN-GP manual: `n_critic=5` pasos de discriminador por 1 de generador, gradient penalty vía `GradientTape` anidado (sin `@tf.function` en la función de penalty para no romper el nesting), Adam D/G separados con los lr/betas de la tabla, **sin** pérdidas auxiliares de bigrama/cycle-time (eso es V5/V6). Se agregó `checkpoint_every=100` — guarda el generador cada 100 epochs en `<output_path>/checkpoints/`. |
| `dg_training.py` | Cuando `model_family == 'simple_gan'`, sobreescribe `epochs=1000, latent_dim=64` y agrega los hiperparámetros de arquitectura V4. |
| `GenerativeGAN/model_prediction/gan_predictor.py` | `load_model` ahora registra `CUSTOM_OBJECTS` del módulo de arquitectura — sin esto, cargar un `.h5` de V4 fallaría porque Keras no reconocería las capas custom. |

**No se tocó** (confirmado compatible): `samples_creator.py`, `dg_prediction.py`
(fuera del punto anterior), `evaluacion/evaluator.py`.

### 10.2 Revisión independiente (etapa 4) — resultado

Se lanzó un agente sin contexto previo de la implementación para revisar la
arquitectura V4 y las métricas RED/CTD/2GD/CONF desde cero. Resultado:

**Arquitectura V4 — confirmada correcta**, incluido el punto más delicado
técnicamente: el nesting de `GradientTape` para el gradient penalty (si estuviera
mal, entrenaría sin errores pero con el gradiente del penalty desconectado — bug
silencioso grave). Encontró y se corrigieron 2 detalles menores:
- Faltaba `use_bias=False` en la proyección categórica del discriminador (`cat_proj`).
- `dropout` no se propagaba desde `GANTrainer` hasta `build_discriminator` (el
  generador sí lo recibía, el discriminador se quedaba en el default de la clase).

**Métricas RED/CTD/2GD — confirmadas correctas** matemáticamente (EMD/Wasserstein-1,
TVD como equivalente exacto del EMD categórico, agregación por caso sin mezclar
trazas).

### 10.3 Bug encontrado y corregido en CONF (código preexistente, `evaluacion/metrics.py`)

El agente marcó como sospechoso (confianza moderada, no pudo ejecutar pm4py) que
`_REVERSE_ARGS = {"altsuccession", "chainsuccession"}` invertía el orden de
argumentos solo para esas 2 plantillas al construir el modelo pm4py desde las reglas
de MINERful.

**Verificación propia, en dos pasos:**
1. Confirmé contra el código fuente real de pm4py (`classic.py::__check_alt_succession`
   / `__check_chain_succession`, vía fetch a GitHub) que pm4py usa un orden **uniforme**
   `(act_couple[0]=activación, act_couple[1]=target)` para TODAS las plantillas
   binarias, sin excepción para alt/chain-succession — descarta la hipótesis original
   de "pm4py tiene semántica invertida para estas 2 plantillas".
2. Pero al releer los CSV de MINERful que ya habíamos visto en esta sesión
   (`modelo_descubierto_peor.csv`, ambas versiones HEAD/origin del merge de
   `recommended_module`), encontré el problema real: **el string `Constraint` de
   MINERful no tiene orden consistente entre familias de plantillas**. Ejemplo real
   del propio repo: `'Precedence(A_SUBMITTED, A_DECLINED)'` con columnas
   `Activation='[A_DECLINED]'`, `Target='[A_SUBMITTED]'` — el primer argumento del
   string es el **target**, no la activación. La familia `Response`/`Succession` sí
   escribe activación-primero (`'Response(A_PARTLYSUBMITTED, A_DECLINED)'` →
   `Activation=[A_PARTLYSUBMITTED]`). El bug real no era "2 plantillas invertidas" —
   era "el código nunca debió parsear el string `Constraint` por posición; MINERful ya
   entrega columnas `Activation`/`Target` sin ambigüedad y el código las ignoraba".

**Fix aplicado**: `_minerful_to_pm4py_model` ahora lee `Activation`/`Target`
directamente (con fallback al parseo posicional del string, sin inversión especial,
si esas columnas no existieran). Se eliminó `_REVERSE_ARGS`.

**Impacto**: cualquier corrida anterior que haya reportado CONF con reglas de la
familia `Precedence` en el conjunto filtrado (soporte≥90%) tiene un CONF
potencialmente sesgado — no confiar en números de CONF calculados antes de este fix
sin volver a correrlos.

**Sigue pendiente** (§8 punto 5): verificación por ejecución real con pm4py — el
razonamiento está verificado contra fuente y contra datos reales del repo, pero ni el
agente revisor ni Claude Code pudieron ejecutar pm4py en este entorno.

### 10.4 Cómo ejecutar (etapa 2 — la corre el usuario, no Claude Code)

Claude Code no tiene TensorFlow/pm4py en su entorno de herramientas — el usuario
ejecuta manualmente en el venv `deep_generator`:

```bash
python dg_training.py -m simple_gan
python dg_prediction.py
```

Ambos ya apuntan a `RunningExample.csv` por defecto. Salidas esperadas:
- `data/1.predicton_models/RunningExample/<run_id>/RunningExample.h5` (generador) +
  `checkpoints/generator_epoch0100.h5`, `...0200.h5`, etc.
- `data/1.predicton_models/RunningExample/<run_id>/parameters/model_parameters.json`
  (incluye ahora `d_model`, `num_heads`, `n_critic`, `gp_lambda`, etc. — documentación,
  no se usan en inferencia).
- `data/4.simulation_results/RunningExample/metricas/metrics_<run_id>.csv` con RED,
  CTD, 2GD, CONF (etapa 3 — ya integrada, no requiere paso manual aparte).

Con 1000 epochs, `n_critic=5` y ~378 trazas de train (70% de 540), el entrenamiento
en CPU debería tomar de minutos a un par de horas — no se ha medido el tiempo real
porque Claude Code no puede ejecutar esta corrida.

---

## 11. GIRO IMPORTANTE (sesión 2026-08-03) — apareció el código real de V2-V6, y el diagnóstico de "degradación" cambia por completo

### 11.1 Contexto: dos carpetas locales del mismo repo

El usuario trabaja con **dos clones locales** del mismo remoto:
- `C:\Users\Diego\OneDrive\Documents\GitHub\...` — donde Claude Code opera (sincronizada
  con OneDrive; ya vimos en §2 que esto causa interferencia con `.git`).
- `C:\Users\Diego\Documents\GitHub\...` — **sin OneDrive**, donde el usuario ejecuta
  manualmente `dg_training.py`/`dg_prediction.py` en el venv `deep_generator` y
  normalmente comitea desde ahí.

El 2026-08-03 se descubrió que la segunda carpeta tenía **~1.480 líneas de trabajo sin
comitear desde julio 2026**, nunca sincronizadas con `origin` ni con la carpeta de
OneDrive. Eso invalida el hallazgo de §4 ("V2-V6 nunca se guardaron en ningún commit")
en su interpretación literal — sí existían, solo que nunca llegaron a git.

### 11.2 Qué había ahí — el V4 real

- `GenerativeGAN/model_training/gan_trainer_v2.py` (`GANTrainerV2`, Jul 8) +
  `GenerativeGAN/model_training/models/model_wgan_transformer.py` (Jul 6): Transformer +
  Time2Vec + WGAN-GP real, **con soporte para las pérdidas auxiliares de bigrama/cycle-time**
  (`bg_lambda`/`ct_lambda`, con relajación suave vía softmax — detalle de implementación
  que no estaba en la tabla de hiperparámetros). Confirmado como la fuente real de la
  tabla V1-V6: `dg_training.py` (esa carpeta) tenía el comentario textual
  `# Configuracion v4 CPU — mejor iteracion (RED=0.020, CTD=2093s, CONF=87.90%)` con
  exactamente los hiperparámetros de §5.
- `dg_prediction.py` mejorado: `_run_gan_pipeline` ahora corre **10 réplicas** de
  generación y guarda `metrics_runs_<run_id>.csv` + `metrics_summary_<run_id>.csv`
  (media/std/min/max) — así es como se produce la tabla que reporta el usuario.
- `GenerativeGAN/model_prediction/gan_predictor.py` mejorado: generación en batch (32
  trazas por llamada al modelo, antes era una por una), guardas NaN/Inf en el
  reescalado, y carga por `SavedModel` (directorio) en vez de `.h5` — diseñado a
  propósito para no necesitar registrar `custom_objects`.
- `evaluacion/metrics.py`: **redefine RED** — de posición temporal relativa (escala
  [0,1], la que verificamos en §10) a posición **ordinal** `(índice/longitud)*100`
  (escala [0,100]). Métrica distinta, no comparable numéricamente con la que dejamos
  en gan-module. **No se pudo confirmar con cuál definición se obtuvo el RED=0.020 de
  referencia** — es una pregunta abierta, ver §11.5.
- El bug de CONF (`Activation`/`Target`, §10.3) **no estaba corregido aquí tampoco** —
  confirmado que es una mejora genuina, independiente de cuál arquitectura se use.

### 11.3 El verdadero diagnóstico de la "degradación" reportada

El usuario compartió métricas (RED=0,686, CTD=791h, 2GD=0,964, CONF=56,15%) pidiendo
diagnosticar por qué eran peores que el V4 de referencia. El diagnóstico inicial
(§10, basado en el log de consola `[GAN] epoch 0000/200 | D: 0.6927...`) identificó
correctamente que corrió la arquitectura V1 vieja, no V4. Pero investigando el
`model_parameters.json` real de esa corrida (`data/1.predicton_models/bpic2012_a/
20260802_.../parameters/model_parameters.json`) aparecieron **dos problemas más**,
independientes de cuál arquitectura se use:

1. **Dataset distinto al de referencia**: la corrida fue sobre **BPI Challenge 2012**
   (`model_type=simple_gan`, actividades `A_ACCEPTED-COMPLETE`, `W_Afhandelen leads-...`),
   no sobre `RunningExample`. Comparar el CTD de esa corrida contra el CTD=2093s de la
   tabla no es válido — son procesos con escalas de tiempo completamente distintas
   (BPIC2012 AS-IS tiene cycle time real de ~78 días según el `CLAUDE.md` del Paper 1).
2. **Bug de preprocesamiento específico de `bpic2012_a`**: `scale_args.dur.max_value=0.0`
   en ese `model_parameters.json`. Con `norm_method=max`, `_rescale()` multiplica por
   `max_value` — si es 0, **toda duración generada se reescala a 0 segundos**,
   sin importar qué tan bien entrenado esté el generador. Y `wait.max_value≈102,8 días`
   con `max_trace_size=175` explica por sí solo un CTD del orden de cientos de horas,
   sin necesidad de invocar mode collapse.
3. Ningún archivo de métricas guardado en disco (`metrics_summary_20260715_040225.csv`,
   `..._085024.csv`, `..._20260802_135044.csv` — este último es el que coincide con lo
   que reportó el usuario) tiene un RED cercano a 0,020. El origen exacto del número de
   referencia de la tabla sigue sin confirmarse en un artefacto guardado.

**Conclusión: no hay evidencia de que V4 sea peor que V1. La comparación que se hizo
no era válida — arquitectura, dataset, y una escala de normalización rota, los tres a
la vez.**

### 11.4 Qué se hizo para reconciliar (mecánica de git, para referencia futura)

1. Commit de resguardo en la carpeta sin OneDrive (`e7d690a`) con todo lo no
   comiteado, **antes** de traer nada de `origin` — para no arriesgar perder ese
   trabajo si el merge salía mal.
2. `git fetch` + `git merge origin/gan-module`. Auto-merge limpio en `dg_training.py`
   y `evaluacion/metrics.py`; conflicto real solo en `gan_predictor.py`.
3. El conflicto de `gan_predictor.py` **no se resolvió "eligiendo un lado"**: mi
   `model_simple_gan.py` y su `model_wgan_transformer.py` cada uno define su propia
   clase `Time2Vec`/`TransformerBlock` con firmas distintas (`dim` vs `output_dim`).
   Registrarlas juntas bajo la misma clave de `custom_objects` haría que una
   sobreescribiera a la otra. Se resolvió separando por **formato de guardado**:
   `SavedModel` (directorio, `GANTrainerV2`/`transformer_wgan`) → sin `custom_objects`
   (como se diseñó originalmente); `.h5` (`GANTrainer`/`simple_gan`) → con las
   `CUSTOM_OBJECTS` de `model_simple_gan.py`.
4. Merge commit `c81f8e8`, pusheado a `origin/gan-module`.
5. Dataset realineado a `RunningExample` en los 3 scripts que estaban desincronizados
   entre sí (`dg_training.py` tenía `bpic2012_a.csv`, `dg_prediction.py` también,
   `dg_boxplot.py` ya tenía `RunningExample` hardcodeado).
6. Carpeta de OneDrive actualizada por fast-forward al mismo commit — las dos carpetas
   locales y `origin/gan-module` quedan sincronizadas en `c81f8e8`.

**No se borró ni descartó nada** — `model_simple_gan.py`/`gan_trainer.py` (mi
reconstrucción V4 bajo `simple_gan`) quedan como ruta secundaria sin usar; la ruta
recomendada de aquí en adelante es `GANTrainerV2`/`transformer_wgan`.

### 11.5 Pendientes abiertos

1. **Confirmar el origen del RED=0,020 de referencia** — ¿con la definición temporal
   (0-1) o la ordinal (0-100)? Ninguna corrida guardada en disco lo aclara. Si
   aparece un artefacto viejo (notebook, captura de consola) que lo confirme, hay que
   actualizar este archivo.
2. **Reentrenar sobre `RunningExample` con `GANTrainerV2`** ahora que todo está
   alineado: `python dg_training.py -m transformer_wgan` (usa las mismas
   configuraciones "V4 CPU" ya hardcodeadas en el bloque `elif model_family ==
   'transformer_wgan':` de `dg_training.py`) seguido de `python dg_prediction.py`.
   Esta sí sería una comparación válida contra la tabla de §5.
3. El bug de `dur.max_value=0.0` es específico del preprocesamiento de `bpic2012_a`
   — no bloquea el punto 2 (que es sobre RunningExample), pero queda pendiente si
   más adelante se retoma BPIC2012 como dataset objetivo.
4. Mismos pendientes de §8 que seguían abiertos: métrica de novedad/diversidad,
   verificación por ejecución real del fix de CONF (§10.3), submódulo `GenerativeLSTM`
   sin commitear.

---

## 12. Cambio de flujo de trabajo — Documents es ahora el repo principal (2026-08-03)

**A partir de ahora, Claude Code trabaja sobre
`C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation`
(sin OneDrive), no sobre la copia de OneDrive.** La copia de OneDrive sigue existiendo
y sincronizada con `origin/gan-module` (por si se necesita consultar), pero ya no es
donde se hacen cambios activos — evita repetir la mecánica de reconciliación de §11.

### 12.1 Primer resultado real de `GANTrainerV2` sobre `RunningExample`

El usuario corrió `python dg_training.py -m transformer_wgan` +
`python dg_prediction.py` (10 réplicas, resumen automático) ya con el dataset
correctamente alineado. Resultado:

| Métrica | Referencia V4 (tabla §5) | Corrida real (media, 10 réplicas) | Lectura |
|---|---|---|---|
| CTD | 2.093 s (0,58 h) | 1.893 s (0,53 h) | Prácticamente igual, incluso mejor |
| 2GD | 0,1273 | 0,1325 | Prácticamente igual |
| RED | 0,0202 (escala **[0,1]**, definición temporal — ver §11.2) | 2,71 (escala **[0,100]**, definición ordinal actual) | **Escalas distintas, no comparables directamente** — normalizando ambas a "% de discrepancia posicional" (0,0202→2,02% vs 2,71→2,71%), son magnitudes muy similares. No hay evidencia de degradación real. |
| CONF | 87,90% | 48,7% (std=0,24pp entre las 10 réplicas — muy estable) | **Único resultado que sí preocupa** — ver §12.2 |

**Conclusión parcial**: 3 de 4 métricas confirman que el pipeline (dataset +
`GANTrainerV2`) está funcionando correctamente y a la altura de la referencia. CONF
es la excepción y requiere diagnóstico aparte antes de sacar conclusiones sobre la
calidad del generador.

### 12.2 CONF bajo — hipótesis y verificación en curso

Argumento a favor de sospechar del **cálculo** de CONF (no del generador): 2GD (que
depende de la misma estructura de orden de actividades en la que se basan la mayoría
de plantillas DECLARE) está casi calcado a la referencia. Sería raro que el generador
capture bien la estructura de bigramas pero falle drásticamente en conformidad de
reglas derivadas de esa misma estructura.

**Diagnóstico propuesto y en curso**: self-check de CONF — comparar
`test_split.csv` contra sí mismo. Como las reglas se minan exigiendo soporte≥90%
sobre ese mismo log, el resultado debería acercarse a ~90%+ casi por construcción.
Si sale bajo también, confirma un problema en el cálculo (posiblemente relacionado
con el fix de `Activation`/`Target` de §10.3, o algo distinto); si sale cerca de 90%,
el 48,7% del GAN es un resultado real del generador a investigar por otro lado.

Script: `verificar_conf_selfcheck.py` (raíz del repo, en la carpeta de Documents —
no comitear la copia de OneDrive si reaparece). Ejecutar con:
```
python verificar_conf_selfcheck.py
```
**Estado: RESUELTO — ver §13.**

---

## 13. CONF corregido (confirmado por ejecución) y RED revertido a la definición correcta (2026-08-03)

### 13.1 Self-check de CONF — bug real confirmado y corregido

El self-check (`test_split.csv` contra sí mismo) dio **48,78%**, matemáticamente
imposible si MINERful y pm4py estuvieran de acuerdo (`dev_fitness` es fraccional;
promediar 41 reglas todas con soporte≥90% individual no puede dar <90%). Se creó
`diagnostico_conf_por_regla.py` para aislar la causa regla por regla — resultado
(datos reales, no hipótesis):

| Familia | Ejemplo | Gap MINERful vs. pm4py |
|---|---|---|
| `response`, `altresponse`, `chainresponse`, `coexistence` | — | 0% (perfecto) |
| `precedence`, `altprecedence`, `chainprecedence`, `succession`, `altsuccession`, `chainsuccession` | `altprecedence('EVENT 7 END','EVENT 1 START')` | **100%** en las 20 reglas de esta familia |
| `init` | `('nan', 'EVENT 1 START')` | 100% (bug distinto) |

**Causa raíz (familia Precedence)**: MINERful usa `Activation` con semántica de
"evento que dispara la verificación" (runtime monitoring) — para `Response` eso es
el evento anterior (coincide con lo que pm4py espera), pero para `Precedence` es el
evento **posterior** (el que, al ocurrir, dispara "¿ya pasó el prerrequisito?").
pm4py en cambio siempre espera `act_couple[0]=prerequisito`. Fix: `_PRECEDENCE_LINEAGE`
en `evaluacion/metrics.py` invierte `(target, activation)` solo para esa familia de 6
plantillas. **Esto invalida por completo el intento de fix del `#10.3` de ayer**
(que asumía, incorrectamente, que ninguna plantilla necesitaba inversión).

**Causa raíz (`init`)**: celdas `Activation` vacías (plantillas unarias) llegan como
`NaN` de pandas; `str(nan) == 'nan'` (string no vacío) rompía la detección de
"unaria". Fix: `_clean_activity` ahora chequea `pd.isna(raw)` primero.

**Confirmado por el usuario tras el fix**: self-check de CONF = **100%** sobre
`test_split.csv`. El cálculo de CONF ahora es correcto y confiable.

### 13.2 RED revertido a la definición temporal (Graziosi et al. 2024)

Al pedirle al usuario que describiera la metodología de las 4 métricas para
verificarla contra el código, se detectó que la definición de RED activa desde el
merge de §11 (posición **ordinal**, `indice/longitud * 100`) no coincide con la
definición del paper de referencia ni con la que el usuario tenía documentada
(posición **temporal**, `(timestamp-inicio)/(fin-inicio)`). Revertido en
`evaluacion/metrics.py::compute_red` a la definición temporal original (escala
`[0,1]`, con la guarda para trazas de instante único). CTD, 2GD y CONF sí coincidían
exactamente con la descripción del usuario — solo RED estaba mal.

### 13.3 Estado actual y siguiente paso

Con ambos fixes (CONF y RED) ya en `origin/gan-module` (commits `47c29d6`,
`36bc33e`), **ningún número de CONF ni de RED reportado hasta ahora en este archivo
es comparable con lo que se obtendría hoy** — todos se calcularon con al menos uno
de los dos bugs presentes. Antes de reportar cualquier conclusión sobre el
desempeño de `GANTrainerV2`, hay que volver a correr la evaluación completa:

```
python dg_training.py -m transformer_wgan
python dg_prediction.py
```

Y comparar las 4 métricas resultantes (ya todas con cálculo correcto) contra la
tabla de referencia de §5 — recordando que el origen exacto de esa tabla de
referencia sigue con la duda abierta de §11.5 (¿bajo qué versión de RED se obtuvo
el 0,0202?). Dado que la definición ya quedó fijada a la temporal (la única
consistente con el paper), esa comparación ahora sí es metodológicamente válida.
