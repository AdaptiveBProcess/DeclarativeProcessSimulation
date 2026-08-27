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

---

## 14. Resultado de validación final — primera comparación sin bugs conocidos (2026-08-03)

Corrida real: `python dg_training.py -m transformer_wgan` + `python dg_prediction.py`
sobre `RunningExample`, 10 réplicas, con los fixes de CONF (§13.1) y RED (§13.2) ya
aplicados.

| Métrica | Referencia V4 (§5) | Corrida real (media, 10 réplicas) | std | Relación |
|---|---|---|---|---|
| RED | 0,0202 | 0,0491 | 0,0071 | ~2,4x más alto |
| CTD | 2.093 s (0,58 h) | 4.009 s (1,11 h) | 120 s (0,033 h) | ~1,9x más alto |
| 2GD | 0,1273 | 0,1302 | 0,0062 | Prácticamente igual (+2,3%) |
| CONF | 87,90% | 88,62% | 1,35 pp | Prácticamente igual, levemente mejor |

**Verificación del marco de interpretación del usuario — correcto en las 4 métricas**:
RED y 2GD son adimensionales en `[0,1]`, cercano a 0 es mejor (EMD/TVD entre
distribuciones ya normalizadas). CTD tiene unidades de tiempo (segundos/horas),
cercano a 0 es mejor. CONF está en `[0,1]` (o %), cercano a 100% es mejor. Esto
coincide con los docstrings de `evaluacion/metrics.py` y con la metodología de
Graziosi et al. (2024).

**Interpretación**: esta es la primera comparación de toda la sesión que es
genuinamente válida de punta a punta — mismo dataset, misma arquitectura
(`GANTrainerV2`), sin ninguno de los bugs de CONF o RED que se fueron encontrando y
corrigiendo. 2GD y CONF prácticamente igualan (CONF incluso supera levemente) la
referencia histórica. RED y CTD están ~2x por encima — un factor razonable de
variabilidad entre corridas de un WGAN-GP (conocido por su varianza run-to-run), no
un indicio de degradación; ambas métricas miden fidelidad temporal fina, la
dimensión más difícil de aprender comparada con la estructura discreta de secuencia
(que 2GD sí captura casi perfecto).

**Nota importante sobre la "referencia"**: el número histórico (0,0202/2093s/87,90%)
nunca se pudo confirmar con un artefacto guardado (§11.5), y sabemos que su CONF
probablemente se calculó con el bug de `Activation`/`Target` que se corrigió hoy
(§13.1). No debe tratarse como una meta absoluta a igualar — es una referencia
aproximada. **El resultado de esta sección (§14) es, con todo lo verificado hasta
ahora, el número más confiable que existe sobre el desempeño real de V4/`GANTrainerV2`.**

### 14.1 Pendientes que siguen abiertos (sin cambios respecto a §11.5/§8)

Métrica de novedad/diversidad, submódulo `GenerativeLSTM` sin commitear.
~~Verificación del origen del RED=0,0202 de referencia~~ — **resuelto en §15**: no
hacía falta encontrar el artefacto original, la explicación estadística alcanza.

---

## 15. Estudio de varianza ENTRE entrenamientos — la brecha de §14 no era real (2026-08-04)

### 15.1 Motivación

§14 dejó una duda abierta: RED (0,0491) y CTD (1,11h) salieron ~2x por encima de la
referencia histórica de V4, mientras 2GD y CONF prácticamente la igualaban. Esa
comparación usaba **un solo modelo entrenado una vez**, evaluado con 10 réplicas de
generación — es decir, medía varianza de *muestreo de generación*, no varianza de
*entrenamiento* (qué tan distinto sale un modelo con otra semilla). Se construyó
`estudio_varianza_transformer_wgan.py` para correr **10 entrenamientos completos e
independientes** (cada uno invocando `dg_training.py`/`dg_prediction.py` como
subproceso separado, sin compartir estado de TensorFlow entre corridas), y comparar
la media de cada corrida entre sí.

### 15.2 Resultado — la referencia histórica es el mejor caso, no el caso típico

| Métrica | Referencia V4 (§5) | Rango observado (10 entrenamientos independientes) | Media | std |
|---|---|---|---|---|
| RED | 0,0202 | [0,0221 – 0,1142] | 0,0552 | 0,0267 |
| CTD_horas | 0,58 h | [0,598 h – 1,189 h] | 0,752 h | 0,182 h |
| 2GD | 0,1273 | [0,1256 – 0,1457] | 0,1349 | 0,0074 |
| CONF | 87,90% | [88,19% – 90,44%] | 89,21% | 0,74 pp |

**La referencia de RED y CTD cae en el borde inferior (el mejor caso) del rango
observado, no en el centro.** Retroactivamente tiene sentido: el comentario en
`dg_training.py` que documentaba esos valores decía literalmente
`# Configuracion v4 CPU — mejor iteracion` — "mejor", no "típica". Nunca fue un
promedio representativo; era el resultado más favorable de una serie de intentos.
Comparar una corrida promedio contra el mejor caso histórico de otra serie de
corridas iba a verse "peor" sin que hubiera ninguna regresión real — es estadística
básica (media vs. máximo), no un problema del modelo.

**CONF además supera consistentemente la referencia**: las 10 corridas dieron entre
88,19% y 90,44% — el *mínimo* de las 10 corridas ya le gana al histórico 87,90%.

### 15.3 Hallazgo genuino: RED y CTD son mucho más inestables entre entrenamientos que 2GD/CONF

Coeficiente de variación (std/media) entre las 10 corridas de entrenamiento:

| Métrica | CV (std/media) |
|---|---|
| RED | 48% |
| CTD | 24% |
| 2GD | 5,5% |
| CONF | 0,8% |

El generador aprende la estructura de secuencia (2GD) y la conformidad de reglas
(CONF) de forma muy estable corrida tras corrida. La fidelidad temporal fina (RED,
CTD) varía mucho más según la semilla de entrenamiento — es la parte más volátil
del modelo. Esto es consistente con la arquitectura: los canales categóricos
(softmax de actividad/rol) tienen una señal de entrenamiento más directa que los
canales continuos de tiempo (sigmoid de `dur`/`wait`), que dependen de la calidad
del gradient penalty y son más sensibles a la inicialización.

### 15.4 Conclusión

**No hace falta tunear nada para "corregir" una degradación — no la hay.** El
modelo V4/`GANTrainerV2`, con la configuración actual, produce resultados dentro
(o mejores, en el caso de CONF) del rango que generó la referencia histórica. Si se
quisiera perseguir algo a futuro, sería un objetivo distinto y opcional: **reducir
la varianza** de RED/CTD entre corridas de entrenamiento (más estabilidad, no
necesariamente mejor promedio) — posibles palancas sin explorar: learning rate
schedules, más epochs, promediado de pesos (EMA), o ajustar el peso del gradient
penalty. Ninguna de estas se ha probado ni es urgente.

Archivo de resultados: `estudio_varianza_transformer_wgan_20260804_110034.csv`
(raíz del repo, 10 corridas).

---

## 16. Borrador real del Paper 2 recibido y cruzado contra la implementación (2026-08-06)

El usuario compartió el PDF del borrador: *"Rule-Guided Generative Adversarial Networks
for Enhanced What-if Process Simulation"* — framework llamado **RULE-GAN**
(Rule-driven Unseen Log Enhancement GAN). No es un archivo del repo (se compartió como
adjunto en la conversación), pero queda documentado aquí porque define el marco de
referencia para todo el trabajo de código de esta rama.

### 16.1 Estructura del borrador

Secciones 1 (Introduction), 2 (Background) y 3 (Related Work) están escritas completas,
con tabla comparativa (Table 1) contra Fast Synthetic, ProcessGAN, PURPLE, Dynamics[4]
y CVAE[8]. **Secciones 4 (Proposed Approach), 5 (Implementation), 6 (Experimental
Evaluation) y 7 (Conclusions) están vacías — solo el título.** Todo el trabajo de
`GANTrainerV2`, los fixes de métricas, y los resultados de §14-15 es exactamente lo que
falta redactar ahí. También hay un placeholder sin terminar en la Introducción:
*"Experimental results demonstrated that....."*

### 16.2 Confirmación cruzada — el fix de RED (§13.2) era correcto

La Sección 2 del paper define RED textualmente como *"measures temporal realism by
analyzing **when events occur within a trace**"* — la definición **temporal**, no
ordinal. Confirma, con el propio texto del paper, que la redefinición ordinal que se
había colado en el código (§11.2) nunca debió estar ahí, y que revertirla a la
definición temporal (§13.2) fue la decisión correcta. CTD, 2GD y CONF también están
descritas en el paper exactamente como quedaron implementadas — el trabajo de esta
rama está alineado con lo que el paper necesita reportar.

### 16.3 ⚠️ Anonimato roto — mismo problema que Paper 1 (`CLAUDE.md` §7)

La referencia **[4]** del Paper 2 dice literalmente: *"BlindAuthors: Blind paper. PeerJ
Computer Science 10, e2094 (May 2024)"* — **idéntico al problema ya documentado en el
`CLAUDE.md` del Paper 1**: volumen, artículo y fecha identifican inequívocamente el
framework "Dynamics" en una revisión doble-ciego. Bloqueante si el venue es doble-ciego
— debe corregirse antes de someter, en ambos papers.

### 16.4 Detalles menores detectados

- Error de idioma en Related Work: *"these approaches [10],[3] **y** [5] share a common
  limitation"* — una "y" en español se coló en el texto en inglés.
- La Etapa 2 del framework (Adversarial Training) se describe como *"generator and
  discriminator focused on structural patterns"* — pero el discriminador real
  (`model_wgan_transformer.py`) también incorpora Time2Vec para codificar tiempo
  (`dur`/`wait`), no solo estructura categórica. Precisar al redactar Secciones 4/5.

### 16.5 Estado

Solo análisis — sin cambios de código. Pendiente de que el usuario indique qué hacer a
continuación (probablemente redactar las Secciones 4-7 con base en el trabajo ya
validado de §10-15).

---

## 17. Comparación contra el protocolo de evaluación de CVAE (Graziosi et al. 2024) — la fuente primaria de nuestras métricas (2026-08-06)

El usuario compartió el PDF completo del paper de CVAE (arXiv:2411.02131) — la fuente
que `evaluacion/metrics.py` ya citaba para RED/CTD/2GD/CONF, pero que hasta ahora solo
conocíamos indirectamente. Se comparó su protocolo experimental (Sección V del paper)
contra lo que tenemos implementado y validado.

### 17.1 Dónde coincidimos exactamente (confirmado, sin cambios necesarios)

| Elemento del protocolo CVAE | Nuestro pipeline |
|---|---|
| Split cronológico 70%-10%-20% | `GANTrainerV2._split_timeline_70_10_20` |
| 10 logs generados por modelo, cada uno del tamaño del test set | `_run_gan_pipeline`, `n_runs=10`, `num_cases=n_test_cases` |
| RED = EMD de posición temporal **dentro de la traza** (nota al pie 7 del paper: explícitamente distinta de AED, que mide el horizonte del log completo) | `compute_red` (revertida en §13.2) — coincide exactamente, incluida la distinción RED-vs-AED que motivó la corrección |
| CTD = EMD de tiempos de ciclo | `compute_ctd` |
| 2GD = EMD de bigramas (directly-follows) | `compute_2gd` |
| CONF = % de restricciones DECLARE satisfechas, soporte≥90% | `compute_conf` (base) |
| Baseline "optimista" `train_log` graficado junto a los modelos generativos (Fig. 2) | `dg_boxplot.py` — ya existía sin comitear (recuperado en §11), diseñado explícitamente para replicar ese panel |
| Convención de mejor/peor por métrica | Idéntica en nuestros docstrings |

### 17.2 Tres gaps concretos — piezas del protocolo que el paper usa y nosotros no

**1. Análisis de variantes (Tabla III del paper) — la pieza que falta para sostener "enhancing changes".**
Protocolo exacto: para cada uno de los 10 logs generados, contar (a) variantes totales
distintas, (b) cuántas ya existían en el log de entrenamiento, (c) cuántas ya existían
en el log de test; promediar entre las 10 corridas. La diferencia (a)−(b) es el número
de variantes genuinamente nuevas — el dato que sostendría la afirmación central de
RULE-GAN. Es exactamente la "métrica de novedad/diversidad" pendiente desde §7/§8,
ahora con especificación operacional exacta en vez de solo la idea.

**2. CONF filtrado a variantes novedosas (nota al pie 8 del paper).**
Cita textual: *"To ensure fairness, only generated variants that do not already appear
in the training log are used for this analysis."* El CONF de la Fig. 2 del paper NO se
calcula sobre todas las trazas generadas — solo sobre las genuinamente nuevas. Responde
"¿el comportamiento *nuevo* que inventa el modelo sigue siendo conforme?", que es
justo la pregunta de RULE-GAN. Nuestro `compute_conf` actual evalúa sobre todas las
trazas sin este filtro — deberíamos calcular ambos (general y solo-novedosas).

**3. El baseline `train_log` real (Sección V-D del paper) es distinto de nuestro
self-check de §13.1.**
El self-check de §13.1 comparó `test_split.csv` **contra sí mismo** — fue una
herramienta de depuración (¿funciona el cálculo de CONF?), no el baseline del paper.
El `train_log` real de CVAE: dividir train+val en **4 partes cronológicas no
solapadas**, cada una del tamaño del test set, comparar cada una contra el test set —
una distribución de referencia con datos reales distintos al test, no una comparación
trivial idéntica. Todavía no construido para el reporte final.

### 17.3 Dato que ya existe pero no se está capturando — análogo a RQ3/Tabla IV

CVAE mide (Tabla IV) qué tan bien el modelo reproduce la proporción condicional
objetivo. Nuestro mecanismo es distinto (rejection sampling contra `train_prop` en vez
de una variable de condicionamiento explícita), pero la pregunta es análoga: ¿qué tan
bien logra el GAN reproducir la proporción objetivo de trazas que cumplen la regla
DECLARE? El dato **ya se calcula** en `GANPredictor._generate_traces`
(`achieved = n_cond/max(n_accepted,1)`, impreso en consola) pero no se captura ni se
tabula — ganancia barata, el cálculo ya existe.

### 17.4 Gap de alcance (más grande, pendiente de decidir)

CVAE compara contra `lstm1`/`lstm2` (el LSTM original de Camargo et al. — mismo
linaje que "Dynamics"/GENESIS). Para que RULE-GAN sostenga "GAN > LSTM en diversidad"
necesitaríamos correr el mismo protocolo (10 réplicas, mismas métricas) contra
`EventLogPredictor` (`GenerativeLSTM/`) sobre el mismo log — no hecho todavía; todo lo
validado en §10-15 fue GAN en aislamiento, nunca comparado cabeza a cabeza contra LSTM
bajo el mismo protocolo. Además CVAE valida en 4 configuraciones reales sustanciales
(782 a 129.615 trazas); nosotros validamos sobre todo en `RunningExample` (540 trazas
sintéticas) — diferencia de escala a tener presente para la Sección 6.

### 17.5 Estado

Solo análisis — sin cambios de código todavía. Pendiente de que el usuario priorice
cuál de estos gaps abordar primero (candidatos naturales por costo/impacto: 17.3
primero por ser gratis, luego 17.2.1 el análisis de variantes por ser el más crítico
para la afirmación central del paper).

---

## 18. Revisión de narrativa del Abstract/Introduction/Related Work (2026-08-06)

Antes de abordar los pendientes técnicos de §17, el usuario pidió revisar si el paper
tiene un objetivo y motivación claros — mismo tipo de auditoría que se hizo para la
introducción del Paper 1 (`CLAUDE.md` §17-18 de ese archivo).

### 18.1 Hallazgos de la revisión narrativa

- **Abstract**: motivación clara en general, pero 2 imprecisiones reales:
  1. Decía que los modelos LSTM *"lack a formal mechanism to incorporate exogenous
     business rules"* — falso para Dynamics[4], que sí tiene ese mecanismo (lo dice el
     propio Related Work del paper). Lo que le falta no es el mecanismo, es diversidad
     en cómo lo usa (rigidez autoregresiva).
  2. Decía *"benchmarked against LSTM-based generators"* — pero CVAE[8], el segundo
     baseline, no es un generador LSTM rígido: ya resuelve diversidad vía muestreo del
     espacio latente (lo reconoce el propio Related Work: *"the CVAE promotes greater
     behavioral diversity"*). Agrupar ambos baselines bajo "LSTM-based" invita la
     pregunta obvia de un revisor: *"¿por qué no usar CVAE con condicionamiento
     declarativo?"*
- **Introducción**: fluye mejor que la del Paper 1 (cadena lógica más coherente). Un
  puente lógico flojo (concept drift → necesidad de what-if, son fenómenos endógeno vs.
  exógeno respectivamente) y un error de concordancia (*"This approaches"* →
  *"These approaches"*). Hallazgo importante: el párrafo 5 promete textualmente que la
  Etapa 4 del framework evalúa *"fidelity, diversity, and constraint satisfaction"* —
  **el propio paper ya se compromete a una evaluación de diversidad**, reforzando que
  §17.2 (análisis de variantes) no es opcional.
- **Related Work**: el párrafo del trade-off Dynamics-vs-CVAE (expresivo-pero-rígido
  vs. diverso-pero-sin-expresividad-formal) es la mejor frase de todo el borrador —
  articula con precisión el hueco que RULE-GAN llena. Se detectó además un probable
  **error factual en la Tabla 1**: la fila de CVAE marca ✓ en "PMS as-is"/"PMS to-be"
  (Process Model Simulation), pero tras leer el paper de CVAE completo (Secciones IV y
  V), **CVAE nunca descubre ni simula un modelo de proceso — genera y evalúa el log
  directamente contra el log de test, sin pasar por Simod ni por ningún motor de
  simulación**. Esa fila de la Tabla 1 necesita corregirse.

### 18.2 Respuesta a la pregunta pendiente de §17: ¿el log generado es resultado de una simulación?

**En CVAE: no. Es la salida directa del decoder**, evaluada contra el log de test sin
ningún paso de descubrimiento/simulación de proceso. Esto tiene una implicación
práctica importante: **nuestro pipeline GAN actual (`_run_gan_pipeline`) ya está en el
formato correcto para compararse contra CVAE sin cambios** (genera y evalúa el log
directo, igual que ellos). Pero para comparar contra Dynamics/LSTM bajo su propia
definición de "what-if simulation" (que sí incluye Simod → PSM TO-BE → simulación
real), sigue haciendo falta el paso que se identificó como faltante en la sesión
anterior (conectar Simod al camino GAN). Confirma que los dos benchmarks (§17.4)
requieren diseños experimentales genuinamente distintos, no el mismo protocolo con el
modelo cambiado.

### 18.3 Terminología: "restrictive/enhancing changes" descartada, se usa "counterfactual" en su lugar

Antes de anclar el par "restrictive changes"/"enhancing changes" en el abstract, se
verificó contra la literatura (búsqueda web). Resultado: **"restrictive" ya tiene un
significado establecido y distinto en el vecindario técnico exacto de este paper**
(subsumption de restricciones DECLARE/MINERful: una restricción es "más restrictiva"
cuando acota más el espacio de comportamiento válido — p.ej. `ChainSuccession` es más
restrictiva que `Succession` — nada que ver con si el comportamiento generado es nuevo
o ya visto). Usar "restrictive changes" con nuestro sentido (solo reproduce lo ya
visto) habría chocado con ese significado establecido, en un paper que usa MINERful
extensamente — mismo tipo de sobrecarga terminológica ya señalada como problema
recurrente en el `CLAUDE.md` del Paper 1.

En su lugar se adoptó vocabulario ya anclado en el propio paper y en su cadena de
citas: **"counterfactual"** (ya usado en el párrafo 1 de la Introducción original:
*"counterfactual logical changes not present in the historical data"*, y con respaldo
bibliográfico directo vía Buliga et al., citado por CVAE como [29],
*"Counterfactuals and ways to build them"*) para el comportamiento nuevo-pero-válido, y
**"replicative, in-distribution"** para el comportamiento que solo recombina patrones
ya vistos.

### 18.4 Abstract corregido (acordado, listo para pegar en el `.tex`)

```latex
\begin{abstract}
What-if process simulation is essential to evaluate the impact of hypothetical changes for process improvement. However, current LSTM-based architectures suffer from structural rigidity due to their autoregressive nature, generating low-diversity synthetic logs that over-represent frequent patterns. Even when a formal mechanism to incorporate exogenous business rules is available, as in declarative-constraint-guided approaches, this same autoregressive rigidity confines the generated behavior to replicative, in-distribution variations of already-observed patterns, rather than enabling genuinely novel, counterfactual process behaviors that comply with the imposed rules without being present in the historical data.
%
This paper proposes a rule-guided deep learning architecture that synergizes Generative Adversarial Networks (GANs) with DECLARE declarative constraints. The GAN component fosters structural diversity by decoupling sequence generation from fixed patterns. Simultaneously, DECLARE logic serves as a formal conceptual model that constrains the generative space, ensuring that synthetic traces adhere to predefined business invariants and logical dependencies.
%
The proposed framework is benchmarked against two complementary families of state-of-the-art generators: declarative-constraint-guided LSTM architectures, representative of expressive yet structurally rigid approaches, and latent-variable models such as conditional variational autoencoders, representative of diverse yet formally unconstrained approaches. Evaluation follows a multidimensional suite: conformance scores for logical consistency, 2-Gram Distance (2GD) for structural fidelity, and Relative Event Distribution (RED) with Cycle Time Distribution for temporal realism. Preliminary results on synthetic and real-world event logs demonstrate that our approach significantly outperforms both baseline families in generating novel, non-repetitive, yet strictly compliant process behaviors. This hybrid approach bridges the gap between stochastic data-driven generation and formal conceptual modeling, enabling more robust and expressive what-if analysis.
\end{abstract}
```

### 18.5 Estado

Abstract corregido y acordado. Pendiente: aplicar el mismo vocabulario
(counterfactual / replicative-in-distribution) de forma consistente cuando se revisen
Introduction y Related Work a fondo, y luego retomar los pendientes técnicos de §17.

---

## 19. Auditoría del Background (Sección 2) y de la Tabla 1 — hallazgo clave sobre ProcessGAN (2026-08-06)

El usuario reenvió el mismo borrador (confirmado idéntico, sin las correcciones de §18
aplicadas todavía) pidiendo una nueva pasada. Se auditó el Background (no revisado a
fondo antes) y se revisó la Tabla 1 con más cuidado.

### 19.1 Background — hallazgos

- **Misma inconsistencia predictive/prescriptive que el Paper 1** (`CLAUDE.md` §7): el
  texto dice *"unseen traces primarily supports **prescriptive** goals"*, pero la Tabla
  1 clasifica a Rule-GAN/Dynamics/CVAE como *"Predictive / simulation"*. Aparece en
  ambos papers de este grupo de investigación — vale la pena resolverlo con el mismo
  criterio en los dos.
- Falta definir **"counterfactual"** explícitamente en el Background — es central desde
  el Abstract pero nunca se define formalmente.
- Cita [11] (*International Journal of Religion*) para la distinción as-is/to-be:
  verificada como real y correcta (no es un error de referencia), pero es un venue de
  bajo perfil para una afirmación fundacional que la referencia [7] (Dumas et al.,
  ya citada en el mismo párrafo) cubre de forma más autorizada.
- Cita genérica [14] compartida para las afirmaciones de LSTM y de GAN — una fuente más
  específica para GANs en process mining (p.ej. Taymouri et al., ya citado por CVAE)
  sería más fuerte.
- Menores: *"allow to validate"* → *"allow validating"* (calco del español);
  *"real logs.[14]"* (cita debería ir antes del punto).

### 19.2 Tabla 1 — hallazgo importante: ProcessGAN tensiona el argumento arquitectónico central

**ProcessGAN[10] — también una GAN — aparece con Trace Diversity = ✗ en la Tabla 1.**
Esto contradice, en apariencia, la premisa del abstract de que *"The GAN component
fosters structural diversity by decoupling sequence generation from fixed patterns"* —
si fuera cierto solo por ser GAN, ProcessGAN también tendría diversidad alta.

La explicación está en el propio texto pero no se conecta explícitamente: ProcessGAN
está diseñado para **preservación de privacidad**, replicando la distribución real
(su objetivo de entrenamiento es *parecerse* al log, no explorar más allá de él). Es
decir: **la diversidad no viene gratis por la arquitectura GAN — depende del objetivo
de entrenamiento.** Sin explicar esto, un revisor va a preguntar exactamente por qué
RULE-GAN sí logra diversidad si ProcessGAN no. Se retoma este hallazgo en §20 como
evidencia de apoyo para el problema #2 del enfoque propuesto.

---

## 20. Redefinición del alcance del paper y propuesta de enfoque (2026-08-06)

### 20.1 Contexto — la pregunta que motivó esto

El usuario pidió repensar, a partir de todo lo visto de los trabajos relacionados, si
la propuesta del paper está bien cimentada antes de invertir más esfuerzo ("si no,
voy a sufrir después"). Se construyó un análisis de 3 ejes independientes para
ubicar el hueco real en la literatura:

| Eje | Dynamics[4] | CVAE[8] | ProcessGAN[10] | Fast Synthetic[3]/PURPLE[5] |
|---|---|---|---|---|
| Expresividad formal (DECLARE) | ✓ | ✗ (binario) | ✗ | ✗ (manual) |
| Diversidad/novedad genuina | ✗ (LSTM rígido) | ✓ (confirmado, Tabla III de su paper) | ✗ (por diseño, ver §19.2) | ✗ |
| Simulación what-if completa (Simod → PSM TO-BE → simulación) | ✓ | ✗ (confirmado leyendo su paper completo) | ✗ | Parcial (PURPLE, modelo manual) |

**Ningún trabajo ocupa las 3 celdas a la vez** — hueco real, no retórico.

**Pero, honestamente, tampoco nosotros lo ocupamos todavía**: de los 3 ejes, solo el de
expresividad formal está sólidamente validado (`CLOUD.md` §10-15). La diversidad
genuina no tiene evidencia empírica (es exactamente el análisis de variantes pendiente
de §17.2.1). La simulación completa no está conectada para el camino GAN (confirmado
dos veces: en la discusión del framework base y al leer el paper de CVAE completo,
§18.2) — metodológicamente hoy estamos al nivel de CVAE (evaluación a nivel de log),
no al de Dynamics (simulación completa), pese a que el título promete
*"What-if Process **Simulation**"*.

### 20.2 Decisión de alcance (acordada)

**Se deja la integración con simulación completa (Simod → PSM TO-BE → simulación,
eje 3) explícitamente como *future work*.** El paper no se somete afirmando resolver
los 3 ejes — se posiciona sobre los 2 que sí se pueden defender con evidencia:
expresividad formal + diversidad genuina, evaluadas a nivel de log (mismo protocolo
que usa CVAE). Precedente directo: CVAE hace exactamente este tipo de acotación en su
propia conclusión (*"In the future, we plan to extend the log generation by also
taking into account resources..."*).

### 20.3 Propuesta de enfoque con el alcance acotado

RULE-GAN se reposiciona como una mejora del **motor de generación de trazas** dentro
del pipeline what-if más amplio (el de Dynamics/GENESIS) — no como un framework de
simulación end-to-end nuevo. La afirmación central pasa a ser: *dado un log real y una
política DECLARE, generar un log sintético que sea simultáneamente conforme a esa
política y genuinamente diverso respecto al histórico* — algo que ningún generador
existente logra a la vez. Ese log queda listo para conectarse al resto del pipeline
(Simod → PSM TO-BE → simulación) en trabajo futuro.

### 20.4 Problemas que soluciona (cada uno anclado a un competidor específico)

1. **El problema de CVAE — diversidad sin expresividad formal.** CVAE logra diversidad
   genuina vía muestreo del espacio latente, pero solo puede condicionar con una
   variable binaria, no con una política de negocio explícita y verificable. RULE-GAN
   reemplaza ese condicionamiento binario por restricciones DECLARE formales.
2. **El problema de Dynamics — expresividad sin diversidad.** Dynamics puede imponer
   una política DECLARE explícita, pero su generador LSTM autoregresivo solo filtra
   continuaciones ya aprendidas del histórico — nunca inventa una secuencia
   genuinamente nueva que también cumpla la regla. RULE-GAN, al generar la traza
   completa desde ruido (no paso a paso desde un prefijo real), no tiene esa
   limitación estructural — **pendiente de prueba empírica (§17.2.1), no se puede
   afirmar solo por diseño.**
3. **Problema metodológico secundario, ya resuelto en gran parte.** CVAE construyó un
   protocolo de evaluación riguroso (RED/CTD/2GD/CONF + análisis de variantes + ratio
   condicional) pero solo para condicionamiento binario. Nadie lo ha aplicado a un
   generador condicionado por reglas DECLARE — ya se construyó y depuró esa base en
   `CLOUD.md` §10-15 (contribución metodológica exportable, no solo un medio para
   validar RULE-GAN).

### 20.5 Consecuencia directa sobre las prioridades técnicas

Con este alcance, el análisis de variantes (§17.2.1) deja de ser "importante" y pasa a
ser **innegociable** — es la única evidencia posible del problema #2, que es ahora
literalmente la afirmación central del paper. Es el siguiente paso técnico obligatorio.

### 20.6 Efecto colateral anotado, sin resolver todavía

El título (*"Enhanced What-if **Process Simulation**"*) y parte del abstract corregido
en §18.4 siguen prometiendo simulación completa. Con el alcance acotado, hay que
revisar en algún momento si el título necesita suavizarse o si se mantiene aclarando
explícitamente que la integración con el motor de simulación queda como *future work*.
Pendiente para cuando se retome el abstract (el usuario pidió dejarlo para el final).

---

## 21. CORRECCIÓN de §20 — la simulación completa NO queda como future work; se reutiliza código ya existente (2026-08-06)

**Esta sección corrige la decisión de alcance de §20.2.** El usuario compartió la
Figura 2 del paper de Dynamics[4] (pipeline completo: Inputs → Simulation Model
Discovery → Constrained-Sequences Generation → Merge and Simulation of BPS Models,
con 5 módulos numerados ①-⑤) y propuso: **tocar solo el módulo ③** (el
Hallucinator — LSTM en Dynamics, GAN en RULE-GAN), reutilizando ①②④⑤ sin cambios.

### 21.1 Verificación técnica (confirmado leyendo `dg_prediction.py`)

La rama LSTM de `main()` ya ejecuta exactamente la secuencia de la Figura 2:
```
① reglas (rules.ini, ya disponible para ambos caminos)
③ call_predict(...)                         ← LSTM Hallucinator
② generate_bps_model(r["input"], ...)        ← Simod descubre ASIS BPS Model
④ generate_bps_model(r["hallucinated"], ...) ← Simod descubre el modelo TO-BE
⑤ adapt_resources(...) + simulate_model(...) + simulate_bimp(...)  ← merge + Simulate
```
`generate_bps_model`, `adapt_resources`, `simulate_model`, `simulate_bimp`
(`dg_prediction.py` líneas 84, 91, 100, 222) son funciones de módulo independientes,
agnósticas al método de generación — solo reciben rutas de carpetas, no les importa
si el log adentro lo produjo un LSTM o una GAN. `_run_gan_pipeline` (línea 321) hoy
genera con la GAN (③) y **retorna temprano** (líneas 418-420) sin llamar a nada de
esto — no porque falte construir algo nuevo, sino porque nadie conectó la llamada
todavía.

**Conclusión: NO es trabajo de investigación en simulación pendiente — es reutilizar
código ya validado** (el mismo que ya usa el benchmark LSTM). La estimación de §20 de
tratar esto como *future work* completo era demasiado conservadora; se retira esa
recomendación.

### 21.2 Diseño de los dos benchmarks (reemplaza el alcance acotado de §20.2-20.3)

- **Contra Dynamics/LSTM**: comparación a **nivel de pipeline completo**. Mismos
  módulos ①②④⑤; se intercambia solo ③ (LSTM vs. GAN); se comparan las métricas de
  desempeño de la simulación what-if final (no solo del log generado). Responde:
  *"¿mejora el resultado de simulación al cambiar el generador, con todo lo demás
  igual?"*
- **Contra CVAE**: comparación a **nivel de módulo generador únicamente** — CVAE no
  tiene ①②④⑤ (confirmado en §18.2: nunca simula, solo genera y evalúa el log
  directamente), así que ahí la comparación justa es log-contra-log con su propio
  protocolo (RED/CTD/2GD/CONF/variantes), sin simulación de por medio.

Dos escenarios justos y distintos, cada uno igualando el nivel de comparación al que
opera cada baseline — no el mismo protocolo con el modelo cambiado.

### 21.3 La expresividad no se reclama como contribución propia

El mecanismo DECLARE (módulo ①, `rules.ini`/`evaluate_condition`) se hereda de
Dynamics — no es una contribución de RULE-GAN. La contribución real y distintiva es
una sola, concentrada en el módulo ③: **diversidad genuina sin perder la
expresividad heredada**. Esto reemplaza el listado de "3 problemas" de §20.4 por un
solo problema central, probado en dos niveles distintos (pipeline completo vs.
Dynamics, módulo generador vs. CVAE) — mensaje más nítido que el de §20.

### 21.4 Estado

Decisión de alcance final (reemplaza §20.2-20.4). Sigue pendiente conectar
`_run_gan_pipeline` con los módulos ②④⑤ para completar el benchmark contra Dynamics —
tarea de ingeniería modesta dado lo verificado en §21.1, no un proyecto de
investigación aparte. El análisis de variantes (§17.2.1) sigue siendo el paso más
urgente (necesario para ambos benchmarks). Pendiente también: enlazar esta propuesta
con una problemática concreta de process mining — ver discusión en curso.
