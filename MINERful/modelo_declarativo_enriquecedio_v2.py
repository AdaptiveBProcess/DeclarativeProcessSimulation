# %%
import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog, Trace, Event 
from pm4py.util import constants
import subprocess
import json
import os
import logging
import re


# 1. Desactivar la bandera global de progreso de PM4Py
constants.SHOW_PROGRESS_BAR = False

# 2. (Opcional) Silenciar los logs informativos para que no salga "exporting..."
logging.getLogger("pm4py").setLevel(logging.WARNING)

def convertir_int(cast_columns_int,df):
    """
    Convierte columnas específicas de un DataFrame al tipo entero de forma segura.

    Utiliza una conversión numérica con redondeo previo para asegurar que representaciones 
    flotantes de IDs o contadores se transformen correctamente. Emplea el tipo 'Int64' 
    de pandas, el cual admite valores nulos (Nullable Integer Data Type).

    Args:
        cast_columns_int (List[str]): Lista con los nombres de las columnas a transformar.
        df (pd.DataFrame): El DataFrame que contiene los datos a procesar.

    Note:
        La función modifica el DataFrame original 'in-place', por lo que no requiere 
        un valor de retorno explícito.
    """
    for i in cast_columns_int:
        # pd.to_numeric con coerce convierte errores en NaN
        # round() asegura que 1.999 sea 2 antes del cast a entero
        # Int64 permite que la columna mantenga NaNs si existen
        df[i]=(
        pd.to_numeric(df[i], errors="coerce")
          .round()
          .astype("Int64")
        )
def convertir_date(cast_columns_date,df):
    """
    Transforma columnas específicas al formato estándar de fecha y hora (datetime).

    Esta función asegura la integridad temporal del log de eventos, convirtiendo 
    representaciones de texto en objetos datetime de pandas. Es un paso esencial 
    para habilitar cálculos de métricas de rendimiento y ordenamiento cronológico.

    Args:
        cast_columns_date (List[str]): Lista de nombres de columnas que contienen 
                                       marcas de tiempo (timestamps).
        df (pd.DataFrame): El DataFrame que representa el log de eventos.

    Note:
        - Utiliza 'errors="coerce"', lo que significa que cualquier valor con 
          formato de fecha inválido se transformará en 'NaT' (Not a Time).
        - La modificación se realiza 'in-place', afectando directamente al objeto original.
    """
    for i in cast_columns_date:
        # pd.to_datetime es el estándar de oro para parsear fechas de forma flexible
        df[i]=(
        pd.to_datetime(df[i], errors="coerce")
        )

def carga_df_to_format_xes(dfa,log_filtrado):
    """
    Exporta un DataFrame al formato estándar XES con metadatos enriquecidos.

    Esta función prepara los datos para herramientas externas de Process Mining, 
    asegurando que las columnas críticas tengan el tipado correcto, manejando valores 
    nulos en atributos organizacionales y configurando las extensiones y 
    clasificadores globales del estándar XES.

    Args:
        df (pd.DataFrame): DataFrame que contiene el log de eventos procesado.
        log_filtrado (str): Ruta y nombre del archivo de salida (ej. 'best_case.xes').

    Note:
        Se utiliza un diccionario de actualización (.update()) para los clasificadores 
        para garantizar la compatibilidad con las estructuras internas de objetos de PM4Py.
    """
    # 1. Preparación de datos y tipado
    df_xes = dfa
    df_xes['case:concept:name'] = df_xes['case:concept:name'].astype(str)
    df_xes['concept:name'] = df_xes['concept:name'].astype(str)
    # Utilidad de PM4Py para asegurar que los timestamps sean compatibles con XES
    df_xes = dataframe_utils.convert_timestamp_columns_in_df(df_xes)
    # Manejo de nulos en atributos organizacionales para evitar errores de esquema
    cols_nan = ['resourceId', 'org:resource', 'resourceCost']
    for col in cols_nan:
        if col in df_xes.columns:
            df_xes[col] = df_xes[col].fillna("nan").astype(str)

    # 2. Conversión a objeto Event Log
    log = pm4py.convert_to_event_log(df_xes, show_progress=False)

    # 3. Configuración de Metadatos y Extensiones Globales (Estándar XES)
    log.attributes['concept:name'] = "Proceso Filtrado"
    log.extensions['Concept'] = {'prefix': 'concept', 'uri': 'http://www.xes-standard.org/concept.xesext'}
    log.extensions['Lifecycle'] = {'prefix': 'lifecycle', 'uri': 'http://www.xes-standard.org/lifecycle.xesext'}
    log.extensions['Organizational'] = {'prefix': 'org', 'uri': 'http://www.xes-standard.org/org.xesext'}

    # Inyección de atributos omni-presentes (requeridos para cumplimiento de esquema)    log.omni_present_trace_attributes = [{'concept:name': '__INVALID__'}]
    log.omni_present_event_attributes = [
        {'concept:name': '__INVALID__'},
        {'lifecycle:transition': 'complete'}
    ]

    # 4. Definición de Clasificadores
    # Los clasificadores permiten a otras herramientas agrupar eventos por nombre o recurso
    nuevos_clasificadores = {
        'Event Name': ['concept:name'],
        'MXML Legacy Classifier': ['concept:name', 'lifecycle:transition'],
        'Resource': ['org:resource']
    }
    log.classifiers.update(nuevos_clasificadores)
    # 5. Escritura final del archivo
    pm4py.write_xes(log,log_filtrado, show_progress=False, parameters={"show_progress": False})
    #print("¡Éxito total! Archivo generado con clasificadores y metadatos.")

def ejecutar_minerful(input_log, output_csv, support=0.1, confidence=0.8):
    """
    Ejecuta el algoritmo MINERful para descubrir restricciones DECLARE a partir de un log XES.

    Esta función actúa como una interfaz entre Python y la implementación original en Java 
    de MINERful. Automatiza la configuración del classpath, la ejecución del subproceso 
    del sistema y la posterior ingesta de las reglas descubiertas en un DataFrame.

    Args:
        input_log (str): Ruta al archivo .xes de entrada (ej. mejor_caso.xes).
        output_csv (str): Ruta donde se guardará temporalmente el resultado de las reglas.
        support (float): Umbral mínimo de soporte (0.0 a 1.0). Por defecto 0.1.
        confidence (float): Umbral mínimo de confianza (0.0 a 1.0). Por defecto 0.8.

    Returns:
        Optional[pd.DataFrame]: Un DataFrame con las reglas DECLARE descubiertas, 
                                o None si ocurre un error en la ejecución.

    Note:
        Requiere que 'MINERful.jar' y su carpeta 'lib' estén en el directorio raíz. 
        Detecta automáticamente el separador de rutas según el sistema operativo (Windows/Linux).
    """
    minerful_jar = "MINERful.jar"
    lib_folder = "lib/*"
    
    # Detector de separador de rutas (Windows usa ; y Linux/Mac usa :)
    path_separator = ";" if os.name == 'nt' else ":"
    classpath = f"{minerful_jar}{path_separator}{lib_folder}"

    # 2. Construir el comando
    # Equivalente a: java -cp ... minerful.MinerFulMinerStarter ...
    cmd = [
        "java", 
        "-cp", classpath,
        "minerful.MinerFulMinerStarter",
        "-iLF", input_log,
        "-s", str(support),
        "-c", str(confidence),
        "-oCSV", output_csv  
    ]

    #print(f"Ejecutando MINERful con soporte {support}...")
    
    # 3. Ejecución del subproceso
    try:
        resultado = subprocess.run(cmd, capture_output=True, text=True, check=True)
        #print("MINERful terminó con éxito.")
        #print(cmd)
        # 4. Leer el resultado en Python
        if os.path.exists(output_csv):
            df=pd.read_csv(output_csv,sep=";", quotechar="'")
            # 1. Limpiar los nombres de las columnas por si traen comillas del CSV
            df.columns = [c.strip("'").strip() for c in df.columns]
            return df
        else:
            print("Error: No se generó el archivo de salida.")
            return None

    except subprocess.CalledProcessError as e:
        print("Error al ejecutar Java:")
        print(e.stderr)
        return None

def generar_analisis_por_traza(df):
    """
    Realiza un análisis multidimensional de rendimiento a nivel de traza individual.

    Esta función integra el descubrimiento de variantes, el cálculo de tiempos de 
    espera/procesamiento y la determinación del Process Cycle Efficiency (PCE). 
    Es el pilar estadístico para el filtrado diferencial entre el mejor y peor caso.

    Args:
        df (pd.DataFrame): Log de eventos con columnas 'case:concept:name', 
                           'concept:name', 'time:timestamp' y 'lifecycle:transition'.

    Returns:
        pd.DataFrame: Reporte detallado por traza incluyendo ID de variante, 
                      tiempos totales, tiempos promedio por actividad y métrica PCE.
    """
    # 1. Preparación y Orden
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])
    
    # 2. CREACIÓN DEL DICCIONARIO DE VARIANTES (Ranking por frecuencia)
    # Identificamos la ruta de cada traza
    df_path = df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x)).reset_index()
    df_path.columns = ['case:concept:name', 'camino_ruta']
    
    # Contamos frecuencia para asignar IDs (La más común será Variant 1)
    variant_ranking = df_path['camino_ruta'].value_counts().reset_index()
    variant_ranking.columns = ['camino_ruta', 'count']
    variant_ranking['variante_id'] = [f"Variant {i+1}" for i in range(len(variant_ranking))]
    
    # Unimos para tener el ID en cada traza
    df_variants_mapped = df_path.merge(variant_ranking[['camino_ruta', 'variante_id']], on='camino_ruta')

    # 3. Cálculo de Tiempos a nivel Actividad (Instancias)
    df_temp = df.copy()
    df_temp['instance'] = df_temp.groupby(['case:concept:name', 'concept:name', 'lifecycle:transition']).cumcount()

    df_pivoted = df_temp.pivot_table(
        index=['case:concept:name', 'concept:name', 'instance'],
        columns='lifecycle:transition',
        values='time:timestamp',
        aggfunc='first'
    ).reset_index()

    df_pivoted = df_pivoted.sort_values(by=['case:concept:name', 'start'])

    # Cálculo de Predecesor para Esperas
    df_pivoted['prev_complete'] = df_pivoted.groupby('case:concept:name')['complete'].shift(1)
    mask_parallel = df_pivoted['start'] == df_pivoted['start'].shift(1)
    df_pivoted.loc[mask_parallel, 'prev_complete'] = df_pivoted['prev_complete'].shift(1)

    # Tiempos por actividad
    df_pivoted['waiting_m'] = (df_pivoted['start'] - df_pivoted['prev_complete']).dt.total_seconds() / 60
    df_pivoted['processing_m'] = (df_pivoted['complete'] - df_pivoted['start']).dt.total_seconds() / 60
    df_pivoted[['waiting_m', 'processing_m']] = df_pivoted[['waiting_m', 'processing_m']].fillna(0)

    # 4. Agrupación por TRAZA (Totales y Promedios)
    df_traza_metrics = df_pivoted.groupby('case:concept:name').agg({
        'waiting_m': ['sum', 'mean'],
        'processing_m': ['sum', 'mean']
    }).reset_index()
    
    df_traza_metrics.columns = [
        'case:concept:name', 
        'tiempo_espera_x_traza', 'tiempo_espera_prom_por_actividad',
        'tiempo_procesamiento_x_traza', 'tiempo_proces_prom_por_actividad'
    ]

    # 5. Cálculo del Tiempo de Ciclo
    df_ciclo = df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max']).reset_index()
    df_ciclo['tiempo_ciclo_x_traza'] = (df_ciclo['max'] - df_ciclo['min']).dt.total_seconds() / 60

    # 6. Unificación Final
    df_final = df_variants_mapped.merge(df_ciclo[['case:concept:name', 'tiempo_ciclo_x_traza']], on='case:concept:name')
    df_final = df_final.merge(df_traza_metrics, on='case:concept:name')

    # 7. Cálculo de PCE y Formateo
    df_final['PCE'] = (df_final['tiempo_procesamiento_x_traza'] / df_final['tiempo_ciclo_x_traza']) * 100
    df_final['PCE'] = df_final['PCE'].replace([np.inf, -np.inf], 0).fillna(0)


    # 8. Cálculo de IL y Formateo
    df_final['IL'] = (df_final['tiempo_espera_x_traza'] / df_final['tiempo_ciclo_x_traza']) * 100
    df_final['IL'] = df_final['IL'].replace([np.inf, -np.inf], 0).fillna(0)

    df_final = df_final.rename(columns={'case:concept:name': 'traza', 'variante_id': 'variante'})
    
    cols_num = [
        'tiempo_ciclo_x_traza', 'tiempo_espera_x_traza', 'tiempo_procesamiento_x_traza',
        'tiempo_espera_prom_por_actividad', 'tiempo_proces_prom_por_actividad', 'PCE'
    ]
    df_final[cols_num] = df_final[cols_num].round(2)

    # Reordenar columnas para la entrega final
    orden = [
        'traza', 'variante', 'tiempo_ciclo_x_traza', 
        'tiempo_espera_x_traza', 'tiempo_procesamiento_x_traza',
        'tiempo_espera_prom_por_actividad', 'tiempo_proces_prom_por_actividad', 
        'PCE','IL'
    ]
    
    return df_final[orden].sort_values(by='variante')

def cargar_csvs_de_carpeta(ruta_carpeta):
    """
    Lee todos los archivos .csv de una carpeta específica y los 
    une en un único DataFrame de pandas.
    """
    # 1. Verificar si la carpeta existe
    if not os.path.exists(ruta_carpeta):
        print(f" La carpeta '{ruta_carpeta}' no existe.")
        return None

    # 2. Listar solo los archivos que terminan en .csv
    archivos_csv = [f for f in os.listdir(ruta_carpeta) if f.endswith('.csv')]
    
    if not archivos_csv:
        print(" No se encontraron archivos .csv en la carpeta.")
        return None

    print(f" Archivos encontrados: {archivos_csv}")

    # 3. Leer y acumular los dataframes en una lista
    lista_df = []
    for archivo in archivos_csv:
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        try:
            # Puedes ajustar el separador (sep=';') según tus archivos
            df_temp = pd.read_csv(ruta_completa)
            lista_df.append(df_temp)
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")

    # 4. Concatenar todos los dataframes en uno solo
    df_final = pd.concat(lista_df, ignore_index=True)
    
    print(f" Éxito: Se cargaron {len(archivos_csv)} archivos en el DataFrame.")
    return df_final


# =============================================================================
# CORRECCIÓN DE LIFECYCLE INCOMPLETO
# Identifica actividades sin mínimo un 'start' Y un 'complete' y completo
# la transición faltante con el mismo timestamp de la existente:
#   - Solo 'start'    → se agrega un 'complete' con el mismo time:timestamp
#   - Solo 'complete' → se agrega un 'start'    con el mismo time:timestamp
# =============================================================================

def diagnosticar_lifecycle_incompleto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un DataFrame con las (caso, actividad) que no tienen ambas
    transiciones 'start' y 'complete', incluyendo cuál es la faltante.
    """
    resumen = (
        df
        .groupby(["case:concept:name", "concept:name"], as_index=False)["lifecycle:transition"]
        .agg(lambda x: ",".join(sorted(x.astype(str))))
        .rename(columns={"lifecycle:transition": "transitions_concat"})
    )

    def clasificar(transitions_str):
        partes = set(transitions_str.split(","))
        tiene_start    = "start"    in partes
        tiene_complete = "complete" in partes
        if tiene_start and not tiene_complete:
            return "falta_complete"
        elif tiene_complete and not tiene_start:
            return "falta_start"
        return "ok"

    resumen["estado"] = resumen["transitions_concat"].apply(clasificar)
    incompletos = resumen[resumen["estado"] != "ok"].copy()
    return incompletos


def corregir_lifecycle_incompleto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega filas sintéticas para actividades con lifecycle incompleto.

    Para cada (caso, actividad) que le falta 'complete':
        toma el registro 'start' existente y duplica con lifecycle='complete'.
    Para cada (caso, actividad) que le falta 'start':
        toma el registro 'complete' existente y duplica con lifecycle='start'.

    El DataFrame resultante queda ordenado por case:concept:name y time:timestamp.
    """
    incompletos = diagnosticar_lifecycle_incompleto(df)

    if incompletos.empty:
        #print("[lifecycle] Todas las actividades tienen 'start' y 'complete'. Sin correcciones.")
        return df.copy()

    n_casos       = incompletos["case:concept:name"].nunique()
    n_actividades = len(incompletos)
    #print(f"[lifecycle] Actividades con lifecycle incompleto: {n_actividades} "
          #f"en {n_casos} caso(s) únicos.")
    #print(incompletos.to_string(index=False))

    filas_sinteticas = []

    # --- Actividades sin 'complete' (solo tienen 'start') ---
    sin_complete = incompletos[incompletos["estado"] == "falta_complete"]
    if not sin_complete.empty:
        claves = set(zip(sin_complete["case:concept:name"], sin_complete["concept:name"]))
        mask = df.apply(
            lambda r: (r["case:concept:name"], r["concept:name"]) in claves
                      and r["lifecycle:transition"] == "start",
            axis=1,
        )
        ref_rows = (
            df[mask]
            .groupby(["case:concept:name", "concept:name"], group_keys=False)
            .apply(lambda g: g.iloc[[0]])
        ).copy()
        ref_rows["lifecycle:transition"] = "complete"
        filas_sinteticas.append(ref_rows)
        #print(f"  → {len(ref_rows)} fila(s) 'complete' sintética(s) agregadas.")

    # --- Actividades sin 'start' (solo tienen 'complete') ---
    sin_start = incompletos[incompletos["estado"] == "falta_start"]
    if not sin_start.empty:
        claves = set(zip(sin_start["case:concept:name"], sin_start["concept:name"]))
        mask = df.apply(
            lambda r: (r["case:concept:name"], r["concept:name"]) in claves
                      and r["lifecycle:transition"] == "complete",
            axis=1,
        )
        ref_rows = (
            df[mask]
            .groupby(["case:concept:name", "concept:name"], group_keys=False)
            .apply(lambda g: g.iloc[[0]])
        ).copy()
        ref_rows["lifecycle:transition"] = "start"
        filas_sinteticas.append(ref_rows)
        #print(f"  → {len(ref_rows)} fila(s) 'start' sintética(s) agregadas.")

    df_corregido = pd.concat([df] + filas_sinteticas, ignore_index=True)
    df_corregido = df_corregido.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).reset_index(drop=True)

    return df_corregido












