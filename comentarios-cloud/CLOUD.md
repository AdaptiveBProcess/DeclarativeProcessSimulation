# CLOUD.md — Bitácora de sesión (rama `gan-module`)

> Contexto persistente para Claude Code sobre el trabajo hecho en esta rama.
> Leer completo al inicio de cada sesión que retome este tema.
> Última actualización: 2026-08-01.

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
   versiones (V1-V6) que el usuario aportó. **Hallazgo crítico**: el código en el repo
   solo implementa la arquitectura V1; las versiones V2-V6 (incluida V4, la
   seleccionada) nunca se comitearon — solo existen como configuración documentada.

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
| `GenerativeGAN/model_training/models/model_simple_gan.py` | Arquitectura generador/discriminador (**hoy: solo V1**) |
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

## 4. HALLAZGO CRÍTICO — el código solo implementa V1, no V4 (la seleccionada)

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

## 8. Próximos pasos (pendientes, ninguno iniciado)

1. **Reconstruir V4 en código** (`model_simple_gan.py` + `gan_trainer.py`):
   Transformer + Time2Vec + WGAN-GP + gradient penalty, siguiendo la especificación
   exacta de §5. Sin esto, `dg_training.py -m simple_gan` sigue entrenando V1.
2. **Diseñar métrica de novedad/diversidad** (§7) si se va a defender la afirmación
   de "comportamiento no visto" con evidencia, no solo por diseño arquitectónico.
3. Decidir qué hacer con los cambios sueltos del submódulo `GenerativeLSTM` (§2,
   detached HEAD, sin commit) — no bloquea lo anterior pero sigue pendiente.

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
