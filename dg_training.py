# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:27:58 2025

@author: David Sequera
"""
import os
import sys
import getopt
import pandas as pd
from pathlib import Path
from support_modules.predictor_adapter import get_latest_output_folder
# from get_folder import ReturnFolderName
from GenerativeLSTM.model_training import model_trainer as tr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _diagnosticar_lifecycle_incompleto(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna las (caso, actividad) que no tienen ambas transiciones 'start' y 'complete'."""
    resumen = (
        df
        .groupby(["case:concept:name", "concept:name"], as_index=False)["lifecycle:transition"]
        .agg(lambda x: ",".join(sorted(x.str.upper().astype(str))))
        .rename(columns={"lifecycle:transition": "transitions_concat"})
    )

    def clasificar(s):
        partes = set(s.split(","))
        if "START" in partes and "COMPLETE" not in partes:
            return "falta_complete"
        elif "COMPLETE" in partes and "START" not in partes:
            return "falta_start"
        return "ok"

    resumen["estado"] = resumen["transitions_concat"].apply(clasificar)
    return resumen[resumen["estado"] != "ok"].copy()


def _corregir_lifecycle_incompleto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega filas sintéticas para actividades con lifecycle incompleto:
      - Solo 'START'    → duplica la fila con lifecycle='COMPLETE' y mismo timestamp.
      - Solo 'COMPLETE' → duplica la fila con lifecycle='START'    y mismo timestamp.
    """
    incompletos = _diagnosticar_lifecycle_incompleto(df)
    if incompletos.empty:
        return df.copy()

    lc_upper = df["lifecycle:transition"].str.upper()
    filas_sinteticas = []

    sin_complete = incompletos[incompletos["estado"] == "falta_complete"]
    if not sin_complete.empty:
        claves = set(zip(sin_complete["case:concept:name"], sin_complete["concept:name"]))
        mask = (
            df.apply(lambda r: (r["case:concept:name"], r["concept:name"]) in claves, axis=1)
            & (lc_upper == "START")
        )
        ref = (
            df[mask]
            .groupby(["case:concept:name", "concept:name"], group_keys=False)
            .apply(lambda g: g.iloc[[0]])
        ).copy()
        ref["lifecycle:transition"] = "COMPLETE"
        filas_sinteticas.append(ref)

    sin_start = incompletos[incompletos["estado"] == "falta_start"]
    if not sin_start.empty:
        claves = set(zip(sin_start["case:concept:name"], sin_start["concept:name"]))
        mask = (
            df.apply(lambda r: (r["case:concept:name"], r["concept:name"]) in claves, axis=1)
            & (lc_upper == "COMPLETE")
        )
        ref = (
            df[mask]
            .groupby(["case:concept:name", "concept:name"], group_keys=False)
            .apply(lambda g: g.iloc[[0]])
        ).copy()
        ref["lifecycle:transition"] = "START"
        filas_sinteticas.append(ref)

    df_corregido = pd.concat([df] + filas_sinteticas, ignore_index=True)
    return df_corregido.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).reset_index(drop=True)


def preprocess_lifecycle_log(input_folder: str, filename: str) -> str:
    """
    Convierte un log con columnas lifecycle:transition y time:timestamp en un
    log con una fila por actividad y columnas start_timestamp / end_timestamp.

    - Toma el timestamp de la fila con lifecycle:transition == 'start' → start_timestamp
    - Toma el timestamp de la fila con lifecycle:transition == 'complete' → end_timestamp
    - Renombra: case:concept:name → caseid, concept:name → task, org:resource → user
    - Guarda el resultado en una carpeta separada (<input_folder>_preprocessed)
      con el MISMO nombre de archivo, para NO modificar el log original.
    - Devuelve la ruta a la carpeta donde quedó el archivo preprocesado.
    """
    csv_path = Path(input_folder) / filename
    df = pd.read_csv(csv_path)

    # Si ya tiene start_timestamp y end_timestamp no necesita preprocesamiento
    if 'start_timestamp' in df.columns and 'end_timestamp' in df.columns:
        print(f"[preprocess] '{filename}' ya tiene start/end timestamp — sin cambios.")
        return input_folder

    # Parsear timestamps: utc=True maneja zonas horarias (+00:00, etc.)
    # tz_convert(None) convierte a UTC y elimina la info de timezone para que
    # log_reader pueda parsear con el formato estándar sin offset.
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True).dt.tz_convert(None)
    df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)

    # Completar actividades que solo tienen 'start' o solo 'complete'
    df = _corregir_lifecycle_incompleto(df)

    # Comparación case-insensitive: acepta 'start'/'START' y 'complete'/'COMPLETE'
    lc = df['lifecycle:transition'].str.upper()
    df_start = df[lc == 'START'].copy()
    df_end   = df[lc == 'COMPLETE'].copy()

    # Emparejar ocurrencias por orden de aparición dentro de (caso, actividad)
    df_start['_occ'] = df_start.groupby(['case:concept:name', 'concept:name']).cumcount()
    df_end['_occ']   = df_end.groupby(['case:concept:name', 'concept:name']).cumcount()

    df_merged = pd.merge(
        df_start[['case:concept:name', 'concept:name', 'org:resource', 'time:timestamp', '_occ']],
        df_end[['case:concept:name', 'concept:name', 'time:timestamp', '_occ']],
        on=['case:concept:name', 'concept:name', '_occ'],
        suffixes=('_start', '_end'),
    )

    df_result = df_merged.rename(columns={
        'case:concept:name':    'caseid',
        'concept:name':         'task',
        'org:resource':         'user',
        'time:timestamp_start': 'start_timestamp',
        'time:timestamp_end':   'end_timestamp',
    }).drop(columns=['_occ'])

    # NaN → 'UNKNOWN' (algunos eventos START no tienen recurso asignado)
    # astype(str) normaliza IDs numéricos (10913.0 → '10913.0') a string
    df_result['user'] = df_result['user'].fillna('UNKNOWN').astype(str)
    df_result = df_result.sort_values(['caseid', 'start_timestamp']).reset_index(drop=True)

    # Guardar en carpeta separada con el MISMO nombre → el log original queda intacto
    out_folder = Path(input_folder).parent / (Path(input_folder).name + '_preprocessed')
    os.makedirs(out_folder, exist_ok=True)
    df_result.to_csv(out_folder / filename, index=False)
    print(f"[preprocess] guardado en '{out_folder / filename}'  "
          f"({len(df_result)} filas, {df_result['caseid'].nunique()} casos) — "
          f"log original sin modificar.")
    return str(out_folder)



def parse_args(argv, filename=''):
    """Parse CLI arguments into a dictionary."""
    defaults = {
        'file_name': filename,
        'model_family': 'lstm',
        'opt_method': 'bayesian',
        'max_eval': 10,
    }
    opt_map = {'-h': 'help', '-f': 'file_name', '-m': 'model_family',
               '-e': 'max_eval', '-o': 'opt_method'}
    try:
        opts, _ = getopt.getopt(argv, "h:f:m:e:o:", list(opt_map.values()))
        for opt, arg in opts:
            key = opt_map.get(opt, opt.lstrip('--'))
            defaults[key] = int(arg) if key == 'max_eval' else arg
    except getopt.GetoptError:
        print("Invalid options.")
        sys.exit(2)
    return defaults


def main(argv):
    FILENAME = 'BPI_Challenge_2012.csv'
    NAME = FILENAME.split('.')[0]
    args = parse_args(argv,filename=FILENAME)

    parameters = {
        'read_options': {
            'timeformat': '%Y-%m-%d %H:%M:%S.%f',
            'column_names': {
                'Case ID': 'caseid',
                'Activity': 'task',
                'lifecycle:transition': 'event_type',
                'Resource': 'user'
            },
            'one_timestamp': False
        },
        'file_name': args['file_name'],
        'opt_method': args['opt_method'],
        'max_eval': args['max_eval'],
        'rp_sim': 0.85,
        'batch_size': 32,
        'norm_method': ['max', 'lognorm'],
        'imp': 1,
        'epochs': 200,
        'n_size': [5, 10, 15],
        'l_size': [50, 100],
        'lstm_act': ['selu', 'tanh'],
        'dense_act': ['linear'],
        'optim': ['Nadam']
    }

    model_type_map = {
        'lstm': ['shared_cat', 'concatenated'],
        'gru': ['shared_cat_gru', 'concatenated_gru'],
        'lstm_cx': ['shared_cat_cx', 'concatenated_cx'],
        'gru_cx': ['shared_cat_gru_cx', 'concatenated_gru_cx'],
        'simple_gan': ['simple_gan']
    }
    model_family = args['model_family']
    parameters['model_type'] = model_type_map.get(model_family, [])
    if 'simple_gan' in parameters['model_type']:
        parameters['gan_pretrain'] = False

    # Preprocesar el log: crear start_timestamp y end_timestamp desde lifecycle:transition
    # El archivo original NO se modifica; el resultado va a una carpeta separada.
    input_folder = f'data/0.logs/{NAME}'
    train_folder = preprocess_lifecycle_log(input_folder, parameters['file_name'])
    # train_folder apunta a la carpeta con el CSV preprocesado (o al original si ya estaba listo)
    # parameters['file_name'] NO cambia → el modelo se sigue llamando RunningExample.h5

    # Los embeddings cacheados pueden quedar desactualizados si el vocabulario de actividades
    # o roles cambia (p.ej. al añadir corrección de lifecycle). Se eliminan para que se
    # regeneren automáticamente con las dimensiones correctas del nuevo log.
    emb_folder = Path(train_folder) / 'embedded_matix'
    if emb_folder.exists():
        import shutil
        shutil.rmtree(emb_folder)
        print(f"[training] Embeddings eliminados para regenerar con vocabulario actual: {emb_folder}")

    # Train the model
    os.makedirs(f'data/1.predicton_models/{NAME}', exist_ok=True)
    tr.ModelTrainer(parameters, input_folder=train_folder, output_folder=f'data/1.predicton_models/{NAME}')
    print(get_latest_output_folder("data/1.predicton_models"))


if __name__ == "__main__":
    main(sys.argv[1:])
