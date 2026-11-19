# CVAE (Graziosi et al., ICPM 2024)

Código, datos y modelos de sus autores: https://github.com/rgraziosi-fbk/cvae-process-mining

## Qué versión se usa

Se usa **la versión del código del paper de ICPM 2024** (commit `c3be4b0`, 08-08-2024).
El 17-09-2024 sus autores agregaron la generación de recursos (`4ea4553`) y el
modelo cambió; los modelos publicados del paper (`additional_material/models/`)
solo cargan con el código anterior a ese cambio. Esa versión no genera recursos.

## Obtener el código

```bash
git clone https://github.com/rgraziosi-fbk/cvae-process-mining.git experimentos/cvae/repo
git -C experimentos/cvae/repo worktree add --detach ../repo_icpm2024 c3be4b0
```

`repo/` queda en `main` (21761e8, 19-05-2025) como referencia; `repo_icpm2024/` es la
versión que se ejecuta, con sus datasets ya preprocesados y divididos y los modelos
entrenados del paper.

## Entorno

```bash
conda create -n cvae python=3.11 -y
pip install -r experimentos/cvae/requirements_windows.txt
```

`requirements_windows.txt` instala solo las dependencias que importa el código de
CVAE, con las versiones fijadas en su `requirements.txt`, salvo cambios obligados por
Windows y Python 3.11:

| Paquete | Su versión | Usada | Motivo |
|---|---|---|---|
| `ray` | 2.24.0 | 2.20.0 (`[tune,train]`) | No existe 2.24.0 para Windows y Python 3.11. 2.20.0 es la más cercana anterior con la API que usan (`ray.train.report`, `get_checkpoint`, `Checkpoint`, `ray.air.config`) |
| dependencias indirectas (p.ej. `yarl`, `psutil`) | fijadas | las que pip resuelva | Sus versiones fijadas no tienen binarios para Windows y Python 3.11 |

`bpic2012_a.pkl` se guardó con una versión vieja de ray (`ray.air.checkpoint`);
`generar.py` reemplaza solo esa clase de metadatos al cargarlo. Los pesos no cambian.

## Generar con sus modelos publicados

```bash
python experimentos/cvae/generar.py --dataset bpic2012_a --corridas 3
```

Por defecto genera tantas trazas como el TEST de CVAE (937 en BPIC2012), todas con la
etiqueta `deviant` (la condición se cumple), como Dynamics y RULE-GAN con
`variation = =1`. `--proporcion` cambia la fracción de trazas `deviant`.

Tiempo de referencia: unos 24 minutos por corrida de 937 trazas en CPU.

## Pendiente

Entrenar CVAE en los logs de Dynamics (RunningExample, PurchasingExample) requiere
convertir el log común a su formato (`scripts/preprocess_log.py`) y agregar la
etiqueta de cada regla; un entrenamiento por regla.
