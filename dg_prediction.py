import support_modules.predictor_adapter as pa
import support_modules.bimp_parser as bp
import os
import getopt
import sys
import gzip
import shutil
import pandas as pd
import datetime
from params.Params import Params
from pathlib import Path
from typing import Optional, Union

from support_modules import traces_evaluation as te


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
        return df

    lc_upper = df["lifecycle:transition"].str.upper()
    df_index = pd.MultiIndex.from_arrays([df["case:concept:name"], df["concept:name"]])
    filas_sinteticas = []

    sin_complete = incompletos[incompletos["estado"] == "falta_complete"]
    if not sin_complete.empty:
        claves = pd.MultiIndex.from_arrays([sin_complete["case:concept:name"], sin_complete["concept:name"]])
        mask = df_index.isin(claves) & (lc_upper == "START")
        ref = (
            df[mask]
            .groupby(["case:concept:name", "concept:name"], group_keys=False)
            .apply(lambda g: g.iloc[[0]])
        ).copy()
        ref["lifecycle:transition"] = "COMPLETE"
        filas_sinteticas.append(ref)

    sin_start = incompletos[incompletos["estado"] == "falta_start"]
    if not sin_start.empty:
        claves = pd.MultiIndex.from_arrays([sin_start["case:concept:name"], sin_start["concept:name"]])
        mask = df_index.isin(claves) & (lc_upper == "COMPLETE")
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


def generate_bps_model(input_folder="log", output_folder="bps", config_file_name="configuration.yaml"):
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    # Crete output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    pa.run_simod_docker(input_path=input_folder, output_path=output_folder, config_file_name=config_file_name)

def simulate_model(input_path, output_path, bpmn_filename, resources_filename, total_cases=100):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)
    latest = pa.get_latest_output_folder(input_path)
    bpmn_path = latest + "/best_result/" + bpmn_filename
    resources_path = latest + "/best_result/" + resources_filename
    pa.run_prosimos_docker(input_path=input_path, output_path=output_path, model_path=bpmn_path, resources_path=resources_path, total_cases=total_cases)

def adapt_resources(original_folder, original_filename, generated_folder, generated_filename, merged_filename):
    original_folder = os.path.join(original_folder, pa.get_latest_output_folder(original_folder) + "/best_result/")
    generated_folder = os.path.join(generated_folder, pa.get_latest_output_folder(generated_folder) + "/best_result/")
    pa.adapt_json(
        asis_bpmn_path= os.path.join(original_folder,original_filename),
        asis_json_path= os.path.join(original_folder,original_filename.replace(".bpmn", ".json")),
        tobe_bpmn_path= os.path.join(generated_folder, generated_filename),
        tobe_json_path=  os.path.join(generated_folder, generated_filename.replace(".bpmn", ".json")),
        merged_json_path= os.path.join(generated_folder, merged_filename)   
    )

def pretty_print_params(params, title="Parameters"):
    print(f"{title}:")
    for key, value in params.items():
        print(f"  {key}: {value}")  


def call_predict(parameters, input_folder="",output_folder="", rules_path="", root_path=""):
    pa.hallucinate(
        parameters=parameters,
        input_folder=input_folder,
        output_folder=output_folder,
        rules_path=rules_path,
        root_path=root_path
    )



def preprocess_xes_log(csv_path: Path, output_folder: Path, filter_sentinels: bool = False) -> Path:
    """
    Convierte el log de eventos al formato esperado por Simod y lo guarda como .csv.gz.
    El CSV original NO se modifica.

    Soporta dos formatos de entrada:

    1. Formato XES (dos filas por actividad con lifecycle:transition):
         case:concept:name  →  caseid
         concept:name       →  task
         org:resource       →  user
         time:timestamp (fila lifecycle:transition == 'start')    →  start_timestamp
         time:timestamp (fila lifecycle:transition == 'complete')  →  end_timestamp

    2. Formato ya procesado (una fila por actividad):
         Columnas caseid, task, user, start_timestamp, end_timestamp ya presentes.
         Solo se seleccionan esas columnas y se comprime.

    Parámetros
    ----------
    csv_path        : ruta al CSV de entrada
    output_folder   : carpeta donde se guarda el .csv.gz resultante
    filter_sentinels: si True, elimina filas donde user ∈ {start,end,Start,End}
                      (útil para limpiar logs alucinados)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Eliminar columnas de índice residuales (Unnamed: 0, etc.)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    if 'start_timestamp' in df.columns and 'end_timestamp' in df.columns:
        # --- Formato ya procesado ---
        df_result = df[['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp']].copy()
        df_result = df_result.sort_values(['caseid', 'start_timestamp']).reset_index(drop=True)
        del df
        print(f"[preprocess_xes_log] Formato estándar detectado — sin pivot necesario.")
    else:
        # --- Formato XES: pivot lifecycle:transition → start_timestamp / end_timestamp ---
        # utc=True + tz_convert(None): normaliza timestamps con timezone (+00:00, etc.)
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
        del df, df_start, df_end, df_merged
        print(f"[preprocess_xes_log] Formato XES detectado — pivot start/complete aplicado.")

    if filter_sentinels:
        _SENTINEL = ['start', 'end', 'Start', 'End']
        df_result = df_result[~df_result['user'].isin(_SENTINEL)]

    os.makedirs(output_folder, exist_ok=True)
    gz_path = Path(output_folder) / (csv_path.stem + '.csv.gz')
    df_result.to_csv(gz_path, index=False, compression='gzip')
    print(f"[preprocess_xes_log] Guardado en: {gz_path}  "
          f"({len(df_result)} filas, {df_result['caseid'].nunique()} casos)")
    return gz_path


def extract_rules(path):
    rules = te.extract_rules(path=path)
    rules = f"{rules['rule']}__"+"__".join(item.replace(' ', '_') for item in rules['path'])
    return rules

def simulate_bimp(input_path="", output_path="", NAME="", PATH="",
                  resources_json=None,
                  num_instances=100,
                  bimp_path="./GenerativeLSTM/external_tools/bimp/qbp-simulator-engine.jar"):
    """
    resources_json: nombre del archivo JSON de recursos dentro de best_result/.
      - ASIS : f"{NAME}.json"         (salida directa de Simod)
      - TOBE : f"{NAME}_merged.json"  (creado por adapt_resources)
      Si no se indica, usa _merged.json por compatibilidad con llamadas anteriores.
    num_instances: número de instancias de proceso que BIMP simulará (default: 100).
    """
    final_input_path = f"{input_path}/{pa.get_latest_output_folder(input_path)}/best_result"
    bpmn_bimp_path = f"{final_input_path}/{NAME}_bimp_version.bpmn"

    if resources_json is None:
        resources_json = f"{NAME}_merged.json"

    # Crear el directorio de salida antes de que BIMP intente escribir el CSV
    os.makedirs(output_path, exist_ok=True)

    bp.embed_qbp_simulation(
        bpmn_path=f"{final_input_path}/{NAME}.bpmn",
        resources_json_path=f"{final_input_path}/{resources_json}",
        bpmn_bimp_path=bpmn_bimp_path,
        exclusive=True,
        num_instances=num_instances,
        )
    pa.run_bimp_docker(
        bimp_path=bimp_path,
        bpmn_path=bpmn_bimp_path,
        csv_path=f"{output_path}/{NAME}_bimp_log.csv",
        path=PATH
    )


def main(argv):
    params = Params(
        root=Path(argv[0]) if argv else Path(),
        log_filename="BPI_Challenge_2012.csv",
    )

    r = params.routes
    sim = params.simulation  # num_instances = total_cases del log original

    rules_name = extract_rules(r["rules"])
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    asis_output = r["simulation"] / f"asis_{run_id}"
    tobe_output = r["simulation"] / f"{rules_name}_{run_id}"

    pretty_print_params(r, title="Routes Paths")
    pretty_print_params(sim, title="Simulation Parameters")
    print(f"Casos en log original: {params.total_cases} | Trazas a simular (Prosimos): {params.tobe_cases}")

    # --- Fase 1 (secuencial): alucinación y preprocesado uno a la vez ---
    # call_predict usa multiprocessing interno — no combinar con otros procesos en 8GB RAM.
    call_predict(sim, r["models"], r["hallucinated"], r["rules"], params.root)
    preprocess_xes_log(r["log"] / params.log_filename, output_folder=r["input"])
    preprocess_xes_log(
        r["hallucinated"] / params.log_filename,
        output_folder=r["hallucinated"],
        filter_sentinels=True,
    )

    # --- Fase 2 (secuencial): descubrir modelos BPS para ASIS y TOBE ---
    # Simod es muy intensivo en memoria — se ejecutan uno tras otro para evitar colapso.
    generate_bps_model(r["input"], r["bps_asis"], "configuration_original.yaml")
    generate_bps_model(r["hallucinated"], r["bps_tobe"], "configuration_generated.yaml")

    # --- Fase 3 (secuencial): adaptar recursos ASIS → TOBE ---
    adapt_resources(
        original_folder=r["bps_asis"],
        original_filename=r["bpmn"],
        generated_folder=r["bps_tobe"],
        generated_filename=r["bpmn"],
        merged_filename=r["merged"],
    )

    # --- Fase 4 (secuencial): ASIS + 3 réplicas TOBE con timestamps distintos ---
    # Cada tobe_output usa run_id + N segundos para garantizar nombres únicos.
    run_id_2 = (datetime.datetime.now() + datetime.timedelta(seconds=1)).strftime("%Y%m%d_%H%M%S")
    run_id_3 = (datetime.datetime.now() + datetime.timedelta(seconds=2)).strftime("%Y%m%d_%H%M%S")
    tobe_output_2 = r["simulation"] / f"{rules_name}_{run_id_2}"
    tobe_output_3 = r["simulation"] / f"{rules_name}_{run_id_3}"

    TOBE_CASES = params.tobe_cases  # número de trazas a simular en Prosimos

    # --- Fase 4 (secuencial): todas las simulaciones una a la vez ---
    # Con 8GB de RAM, cada JVM Docker corre sola para evitar colapso de memoria.
    sim_tasks = [
        ("Prosimos ASIS",      lambda: simulate_model(
            input_path=r["bps_asis"], output_path=asis_output,
            bpmn_filename=r["bpmn"], resources_filename=f"{params.name}.json",
            total_cases=TOBE_CASES,
        )),
        ("Prosimos TOBE rep1", lambda: simulate_model(
            input_path=r["bps_tobe"], output_path=tobe_output,
            bpmn_filename=r["bpmn"], resources_filename=r["merged"],
            total_cases=TOBE_CASES,
        )),
        ("Prosimos TOBE rep2", lambda: simulate_model(
            input_path=r["bps_tobe"], output_path=tobe_output_2,
            bpmn_filename=r["bpmn"], resources_filename=r["merged"],
            total_cases=TOBE_CASES,
        )),
        ("Prosimos TOBE rep3", lambda: simulate_model(
            input_path=r["bps_tobe"], output_path=tobe_output_3,
            bpmn_filename=r["bpmn"], resources_filename=r["merged"],
            total_cases=TOBE_CASES,
        )),
    ]

    for name, fn in sim_tasks:
        try:
            fn()
            print(f"✅ {name} completado")
        except Exception as e:
            print(f"❌ {name} falló: {e}")
   

if __name__ == "__main__":
    main(sys.argv[1:])