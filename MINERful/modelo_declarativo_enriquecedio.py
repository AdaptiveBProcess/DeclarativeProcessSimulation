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

# Calcular la columna de actividades únicas por variante
def extraer_ruta_unica(camino_str):
    """
    Elimina repeticiones consecutivas de actividades en una traza de proceso.

    Esta función procesa una cadena de texto que representa una variante y simplifica
    los auto-bucles inmediatos (self-loops), manteniendo solo una instancia de la 
    actividad cuando esta ocurre de forma sucesiva. 
    Args:
        camino_str (str): Cadena de actividades separadas por el delimitador ' -> '.
                          Ejemplo: "Inicio -> Aprobación -> Aprobación -> Fin"

    Returns:
        str: Cadena normalizada sin repeticiones adyacentes.
             Ejemplo: "Inicio -> Aprobación -> Fin"
    """
    # Descomponemos la cadena en una lista de actividades
    pasos = camino_str.split(' -> ')
    resultado = []
    
    for i in range(len(pasos)):
        # Lógica de filtrado: se agrega el paso solo si es el inicio 
        # o si es distinto al paso inmediatamente anterior.
        if i == 0 or pasos[i] != pasos[i-1]:
            resultado.append(pasos[i])
    # Reconstruimos la cadena con el formato original       
    return ' -> '.join(resultado)

def calculo_variantes(df):
    """
    Realiza el análisis de variantes de un event log para identificar rutas de ejecución y métricas.

    Esta función agrupa los eventos por caso para construir las variantes y calcula estadísticas de 
    rendimiento como frecuencia y tiempo de ciclo promedio.

    Args:
        df (pd.DataFrame): Log de eventos con columnas estándar de Process Mining 
                           ('case:concept:name', 'time:timestamp', 'concept:name').

    Returns:
        pd.DataFrame: Resumen de variantes con columnas: Id, Cantidad_ejecuciones, 
                      Trazas (IDs de casos), Tiempo_ciclo_horas, Ruta_unica y Camino_ruta.
    """
    # 1. Preparación y ordenamiento cronológico por caso
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

    # 2. Cálculo del tiempo de ciclo (Cycle Time) por caso
    case_times = df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max']).reset_index()
    case_times['duration_hrs'] = (case_times['max'] - case_times['min']).dt.total_seconds() / 3600

    # 3. Reconstrucción de la secuencia de actividades (Trazas)
    df_variants = df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x)).reset_index()
    df_variants.columns = ['case:concept:name', 'variant_path']

    # 4. Consolidación de datos de duración y rutas
    df_combined = df_variants.merge(case_times[['case:concept:name', 'duration_hrs']], on='case:concept:name')

    # 5. Agregación estadística por variante (frecuencia y promedio de tiempo)
    variant_analysis = df_combined.groupby('variant_path')['duration_hrs'].agg(['mean', 'count']).reset_index()
    
    # 6. Mapeo de IDs de casos asociados a cada variante
    variant_analysis1 = df_combined.groupby('variant_path')['case:concept:name'].unique().reset_index()
    variant_analysis1.columns = ["variant_path", "Trazas"]
    
    # 7. Limpieza y formateo final del dataset de resultados
    variant_analysis = variant_analysis.merge(variant_analysis1, on='variant_path')
    variant_analysis = variant_analysis.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    # Asignación de identificadores de variante y renombrado de columnas
    variant_analysis['variant_id'] = [f"Variant {i+1}" for i in range(len(variant_analysis))]
    variant_analysis = variant_analysis[['variant_id', 'count', 'mean', 'variant_path', 'Trazas']]
    variant_analysis.columns = ['Id', 'Cantidad_ejecuciones', 'Tiempo_ciclo_horas', 'Camino_ruta', 'Trazas']

    # Aplicación de limpieza de ruta y reordenamiento final
    variant_analysis['Ruta_unica'] = variant_analysis['Camino_ruta'].apply(extraer_ruta_unica)
    variant_analysis = variant_analysis[['Id', 'Cantidad_ejecuciones', 'Trazas', 'Tiempo_ciclo_horas', 'Ruta_unica', 'Camino_ruta']]

    return variant_analysis



def seleccion_variante(variant_analysis):   
    """
    Selecciona las variantes más eficientes basadas en el Process Cycle Efficiency (PCE).

    Esta función implementa el filtrado del 'Mejor Caso' (Best-Case Scenario). 
    Ordena las variantes de forma descendente según su PCE, identificando aquellas 
    con la mayor proporción de tiempo de valor agregado respecto al tiempo de ciclo total.

    Args:
        variant_analysis (pd.DataFrame): DataFrame con el análisis de variantes 
                                         que debe incluir la columna 'PCE'.

    Returns:
        pd.DataFrame: Un subconjunto del DataFrame original con las N variantes 
                      más eficientes, donde N está definido por la variable global 
                      'cantidad_reglas'.
                      
    Note:
        Depende de la variable global 'cantidad_reglas' para determinar el tamaño 
        del corte (top-N).
    """
    # Ordenamos por PCE de mayor a menor (los más eficientes primero)
    # y tomamos las primeras N muestras según el parámetro de configuración.
    global cantidad_reglas
    df_filtrado=variant_analysis.sort_values(by="PCE",ascending=False).head(cantidad_reglas)
    return df_filtrado


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

def calculate_waiting_processing_time(df):
    """
    Calcula los tiempos de espera (waiting) y procesamiento (service) por variante.

    Esta función descompone el tiempo de ciclo total identificando la latencia entre 
    actividades consecutivas. Implementa una lógica de pivotaje para manejar 
    transiciones de ciclo de vida (start/complete) y una corrección para estructuras 
    en paralelo, asegurando que la espera se mida respecto al predecesor real.

    Args:
        df (pd.DataFrame): Log de eventos que debe contener columnas de ciclo de vida 
                           ('lifecycle:transition') con valores 'start' y 'complete'.

    Returns:
        pd.DataFrame: Un resumen por variante ('case:concept:name') con los promedios de 
                      'waiting_m' (espera) y 'processing_m' (procesamiento) en minutos.
    """
    # 1. Ordenamiento cronológico estricto
    df_test = df.sort_values(by=['case:concept:name', 'time:timestamp'])
    # 2. Pivotar para obtener una fila por actividad dentro de cada caso
    # Esto organiza los estados 'start' y 'complete' como columnas

    df_pivoted = df_test.pivot_table(
        index=['case:concept:name', 'concept:name'],
        columns='lifecycle:transition',
        values='time:timestamp',
        aggfunc='first'
    ).reset_index()

    # 3. Asegurar el orden cronológico para identificar predecesores
    df_pivoted = df_pivoted.sort_values(by=['case:concept:name', 'start'])

    # 4. Obtener el fin de la actividad anterior (Predecesor)
    # Se usa shift(1) para traer el 'complete' de la fila de arriba
    df_pivoted['prev_complete'] = df_pivoted.groupby('case:concept:name')['complete'].shift(1)

    # --- LÓGICA PARA TAREAS EN PARALELO (C y D) ---
    # Si dos tareas inician al mismo tiempo, ambas deben compararse contra
    # el final de la tarea que las habilitó 
    mask_parallel = df_pivoted['start'] == df_pivoted['start'].shift(1)
    df_pivoted.loc[mask_parallel, 'prev_complete'] = df_pivoted['prev_complete'].shift(1)

    df_pivoted['waiting_m'] = (df_pivoted['start'] - df_pivoted['prev_complete']).dt.total_seconds() / 60

    # 6. CÁLCULO DEL TIEMPO DE PROCESAMIENTO EN HORAS
    df_pivoted['processing_m'] = (df_pivoted['complete'] - df_pivoted['start']).dt.total_seconds() / 60

    # Limpiar valores nulos (la primera actividad del proceso no tiene espera)
    df_pivoted[['waiting_m', 'processing_m']] = df_pivoted[['waiting_m', 'processing_m']].fillna(0)
    # Mostrar resultado final
    df_pivoted=df_pivoted[["case:concept:name","concept:name","waiting_m","processing_m"]]

    df_pivoted_1 = df_pivoted.groupby('case:concept:name').agg({
        "waiting_m": "mean",
        "processing_m": "mean"
    }).reset_index()
    df_pivoted_1=df_pivoted_1[["case:concept:name","waiting_m","processing_m"]]
    cols = ['waiting_m', 'processing_m']
    df_pivoted_1[cols] = df_pivoted_1[cols].astype(float).round(2)
    return df_pivoted_1
    #df_pivoted_1.to_csv('tiempos-comprobar.csv', index=False, decimal=',', sep=';')
    #df_pivoted_1.head()



def union_metricas(df,waitin_processing_time_df):
    """
    Consolida las métricas de rendimiento temporal con el análisis de variantes.

    Esta función realiza una operación de "vínculo reverso": toma las variantes de 
    proceso, expande sus casos asociados (trazas), integra los tiempos de espera y 
    procesamiento calculados a nivel de instancia, y finalmente vuelve a agregar 
    los resultados para obtener promedios representativos por cada variante.

    Args:
        df (pd.DataFrame): DataFrame de análisis de variantes (contiene la lista de IDs en 'Trazas').
        waitin_processing_time_df (pd.DataFrame): Métricas de tiempo calculadas por caso.

    Returns:
        pd.DataFrame: Un dataset enriquecido y ordenado por frecuencia, que sirve de base 
                      para el cálculo del PCE y el Índice de Latencia.
    """
    # Desanidación de trazas (Explode) 
    # Esto permite que cada variante se conecte con sus casos individuales
    df_union = df.explode('Trazas')

    # 2. Fusión de datos (Join)
    # Se vinculan los tiempos calculados por caso con sus respectivas variantes
    df_final_merged = df_union.merge(
        waitin_processing_time_df, 
        left_on='Trazas', 
        right_on='case:concept:name', 
        how='left'
    )

    # 3. Re-agregación estadística por variante
    # Se calculan los promedios de tiempo para el grupo y se reconstruye la lista de trazas
    resultado = df_final_merged.groupby(['Id', 'Cantidad_ejecuciones', 'Tiempo_ciclo_horas', 'Ruta_unica', 'Camino_ruta']).agg({
        'waiting_m': 'mean',
        'processing_m': 'mean',
        'Trazas': lambda x: list(x)  # Reconstruimos la lista de trazas originales
    }).reset_index()

    # 5. Renombrado semántico de columnas
    # Mejora la legibilidad del reporte final para el analista de negocio
    resultado['tiempo_ciclo_minutos'] = (resultado['Tiempo_ciclo_horas'] * 60).round(2)
    resultado.rename(columns={
        'Trazas': 'trazas',
        'waiting_m': 'tiempo_espera_prom_por_actividad_min',
        'processing_m': 'tiempo_proces_prom_por_actividad_min',
        'Ruta_unica': 'camino_ruta_unica',
        'Camino_ruta': 'camino_ruta',
        'Cantidad_ejecuciones':'cantidad_ejecuciones_por_variante',
        'tiempo_ciclo_minutos':'tiempo_ciclo_prom_por_variante_min',
        'Id':'id'
    }, inplace=True)

    # 6. Estructuración y Ordenamiento
    orden_columnas = [
        'id', 
        'cantidad_ejecuciones_por_variante', 
        'trazas', 
        'tiempo_ciclo_prom_por_variante_min', 
        'tiempo_espera_prom_por_actividad_min', 
        'tiempo_proces_prom_por_actividad_min', 
        'camino_ruta_unica', 
        'camino_ruta'
    ]

    df_resultado_final = resultado[orden_columnas].sort_values(by='cantidad_ejecuciones_por_variante', ascending=False)
    return df_resultado_final

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
        'PCE'
    ]
    
    return df_final[orden].sort_values(by='variante')

def generar_analisis_por_variante (df):
    """
    Orquestador del análisis de rendimiento basado en variantes de proceso.
    Args:
        df (pd.DataFrame): Log de eventos original con marcas de tiempo y 
                           ciclo de vida (start/complete).

    Returns:
        pd.DataFrame: Matriz consolidada de variantes con indicadores clave de 
                      desempeño (KPIs) temporales, utilizada para el filtrado 
                      diferencial de casos.
    """
    #calculo de variantes de traza -> tiempo ciclo
    variant_analysis= calculo_variantes(df)

    #calculo waiting time y processing time para cada traza
    waitin_processing_time_df=calculate_waiting_processing_time(df)

    # analisis por variante
    df_resultado_final=union_metricas(variant_analysis,waitin_processing_time_df)
    return df_resultado_final

def obtener_top_cuellos_botella(df_analisis):
    """
    Calcula el Índice de Latencia (IL) para identificar y extraer las trazas críticas.

    Esta función implementa la métrica de Latency Index para detectar cuellos de botella. 
    A diferencia del PCE, el IL resalta las ejecuciones donde el tiempo de espera 
    representa la mayor parte del tiempo de ciclo. El resultado permite aislar el 
    'Peor Caso' para el posterior filtrado diferencial de reglas DECLARE.

    Args:
        df_analisis (pd.DataFrame): DataFrame generado por 'generar_analisis_por_traza' 
                                    que contiene las métricas temporales base.

    Returns:
        pd.DataFrame: Un subconjunto con las top-N trazas que presentan mayor 
                      ineficiencia temporal (cuellos de botella).

    Note:
        - El IL se calcula como: (tiempo_espera / tiempo_ciclo) * 100.
        - Utiliza la variable global 'cantidad_reglas' para definir el límite del ranking.
    """
    global cantidad_reglas
    #  Crear una copia para no modificar el dataframe original
    df = df_analisis.copy()

    df['IL'] = (df['tiempo_espera_x_traza'] / df['tiempo_ciclo_x_traza'] * 100).fillna(0)

    #  Ranking: Los 5 casos con mayor Latencia (donde la espera pesó más)
    # Si hay empate en IL, desempatamos con el tiempo de espera absoluto más alto
    top_5_cuellos = df.sort_values(by=['IL', 'tiempo_espera_x_traza'], ascending=False).head(cantidad_reglas)

    
    #  Retornar las columnas originales solicitadas más el BTI
    columnas_retorno = [
        'traza', 'variante', 'tiempo_ciclo_x_traza', 
        'tiempo_espera_x_traza', 'tiempo_procesamiento_x_traza',
        'tiempo_espera_prom_por_actividad', 'tiempo_proces_prom_por_actividad', 
        'PCE','IL'
    ]

    return top_5_cuellos[columnas_retorno]

def seleccion_optima_con_tabla(df_analisis):
    """
    Identifica la pareja óptima de trazas (Mejor vs. Peor) maximizando el contraste.

    Esta función evalúa dos estrategias competitivas para seleccionar los casos de estudio:
    1. Maximizar el PCE (Mejor Caso) y buscar el peor contraejemplo en una variante distinta.
    2. Maximizar el IL (Peor Caso) y buscar el mejor contraejemplo en una variante distinta.
    
    El objetivo es encontrar la combinación que presente la mayor suma de métricas 
    extremas (Suma de Contraste), garantizando que las diferencias en el flujo 
    de proceso sean lo más evidentes posible para el motor de descubrimiento.

    Args:
        df_analisis (pd.DataFrame): DataFrame con métricas de rendimiento por traza.

    Returns:
        tuple: (pd.DataFrame con la mejor traza, pd.DataFrame con la peor traza).
    """
    df = df_analisis.copy()
    # Usamos los totales recién calculados
    df_analisis['PCE'] = (df_analisis['tiempo_procesamiento_x_traza'] / df_analisis['tiempo_ciclo_x_traza']) * 100
    df_analisis['PCE'] = df_analisis['PCE'].replace([np.inf, -np.inf], 0).fillna(0)

    # Asegurar que el Índice de Latencia esté calculado
    df['IL'] = (df['tiempo_espera_x_traza'] / df['tiempo_ciclo_x_traza'] * 100).fillna(0)

    # --- OPCIÓN 1: Priorizar Mejor -> Buscar Peor de otra variante ---
    mejor_op1 = df.sort_values(by="PCE", ascending=False).iloc[0]
    df_ex_mejor = df[df['variante'] != mejor_op1['variante']]
    peor_op1 = df_ex_mejor.sort_values(by=['IL', 'tiempo_espera_x_traza'], ascending=False).iloc[0]
    suma_op1 = mejor_op1['PCE'] + peor_op1['IL']

    # --- OPCIÓN 2: Priorizar Peor -> Buscar Mejor de otra variante ---
    peor_op2 = df.sort_values(by=['IL', 'tiempo_espera_x_traza'], ascending=False).iloc[0]
    df_ex_peor = df[df['variante'] != peor_op2['variante']]
    mejor_op2 = df_ex_peor.sort_values(by="PCE", ascending=False).iloc[0]
    suma_op2 = mejor_op2['PCE'] + peor_op2['IL']

    # --- Creación de la Tabla Comparativa ---
    data_tabla = {
        "Estrategia de Selección": ["1. Priorizar Mejor", "2. Priorizar Peor"],
        "ID Mejor Traza": [mejor_op1['traza'], mejor_op2['traza']],
        "PCE (Mejor)": [mejor_op1['PCE'], mejor_op2['PCE']],
        "ID Peor Traza": [peor_op1['traza'], peor_op2['traza']],
        "IL (Peor)": [peor_op1['IL'], peor_op2['IL']],
        "Suma (Contraste)": [suma_op1, suma_op2]
    }   
    tabla_decision = pd.DataFrame(data_tabla)
    
    # --- Elección de la mejor pareja ---
    if suma_op1 >= suma_op2:
        mejor_final = mejor_op1
        peor_final = peor_op1
        ganador = "Opción 1"
    else:
        mejor_final = mejor_op2
        peor_final = peor_op2
        ganador = "Opción 2"
    #print("### TABLA DE DECISIÓN DE TRAZAS ###")
    #print(tabla_decision.to_string(index=False))
    #print(f"\nResultado: Se elige la **{ganador}** por presentar mayor brecha de comportamiento.")

    return pd.DataFrame([mejor_final]), pd.DataFrame([peor_final])

def procesar_modelo_declarativo(ruta_log, ruta_salida, soporte, etiqueta):
    """
    Ejecuta MINERful sobre un log específico y valida el resultado.
    
    Args:
        ruta_log (str): Ruta del archivo XES de entrada.
        ruta_salida (str): Ruta para el CSV de salida.
        soporte (float): Nivel de soporte para el algoritmo.
        etiqueta (str): Nombre descriptivo para el log (ej. 'Mejor Caso').
    """
    #print(f"\n--- Procesando Modelo: {etiqueta} ---")
    
    # Ejecución de la función principal
    datos = ejecutar_minerful(ruta_log, ruta_salida, soporte)
    
    # Validación modular
    if datos is not None and not datos.empty:
        #print(f"Éxito: Se encontraron {len(datos)} restricciones para {etiqueta}.")
        return datos
    else:
        print(f"Advertencia: El archivo CSV para {etiqueta} está vacío o no se generó.")
        return None

def analizar_diferencia_reglas(df_excelencia, df_deficiencia, etiqueta_a="Mejor Caso", etiqueta_b="Peor Caso"):
    """
    Realiza un análisis diferencial entre dos conjuntos de reglas DECLARE.

    Identifica las restricciones de cumplimiento que están presentes en el modelo de 
    excelencia pero ausentes en el de deficiencia, lo que permite proponer 
    recomendaciones de mejora basadas en evidencia.

    Args:
        df_excelencia (pd.DataFrame): Reglas descubiertas en el log optimizado.
        df_deficiencia (pd.DataFrame): Reglas descubiertas en el log ineficiente.
        etiqueta_a (str): Nombre del primer conjunto.
        etiqueta_b (str): Nombre del segundo conjunto.

    Returns:
        set: Conjunto de reglas únicas presentes solo en el modelo de excelencia.
    """
    if df_excelencia is None or df_deficiencia is None:
        print("Error: Uno de los DataFrames es nulo. No se puede realizar la comparación.")
        return set()

    # 1. Extracción de conjuntos (Sets) para operaciones algebraicas
    # Usamos la columna 'Constraint' que es el identificador único de la regla DECLARE
    reglas_a = set(df_excelencia['Constraint'])
    reglas_b = set(df_deficiencia['Constraint'])

    # 2. Cálculo de la diferencia (Análisis Delta)
    # ¿Qué patrones de éxito existen en A que no se replican en B?
    reglas_unicas_excelencia = reglas_a - reglas_b

    # 3. Reporte de hallazgos
    #print(f"\n--- Análisis Diferencial: {etiqueta_a} vs {etiqueta_b} ---")
    #print(f"Reglas recomendadas (exclusivas de {etiqueta_a}): {len(reglas_unicas_excelencia)}")
    
    return reglas_unicas_excelencia
def convertir_declare_to_declarative(regla_input, nombre_archivo="rules_recommended.ini"):
    """
    Convierte regla Declare a formato de notación declarative.
    y guarda el resultado en un archivo .ini
    """
    
    # Diccionario de mapeo basado estrictamente en el archivo fuente proporcionado
    # Se utiliza Case-Insensitive para evitar errores si viene como CoExistence o Coexistence
    mapeo = {
        "absence": "^{A}",           # 
        "required": "{A}",                       # 
        "response": "{A} >> * >> {B}",           # 
        "alternateresponse": "{A} >> {B}",       # 
        "precedence": "{A} >> | >> {B}",         # 
        "succession": "{A} >> # >> {B}",         # 
        "coexistence": "{A} >> + >> {B}",        # 
        "notchainsuccession": "{A} >> - >> {B}"  # 
    }

    # 1. Extraer nombre de función y argumentos (ej. CoExistence y A_DECLINED, A_PARTLYSUBMITTED)
    match = re.match(r"(\w+)\((.*)\)", regla_input)
    if not match:
        return "Error: El formato de la regla no es válido."

    nombre_funcion = match.group(1).lower()
    argumentos = [arg.strip() for arg in match.group(2).split(",")]

    # 2. Aplicar la lógica de conversión
    if nombre_funcion in mapeo:
        formato = mapeo[nombre_funcion]
        
        # Caso especial para Required (1 argumento) o Absence (que según tu fuente usa 2)
        if len(argumentos) == 2:
            path_str = formato.format(A=argumentos[0], B=argumentos[1])
        elif len(argumentos) == 1:
            path_str = formato.format(A=argumentos[0], B="")
        else:
            path_str = "ERROR_ARGUMENTOS"
    else:
        path_str = "FUNCION_NO_MAPEADA"

    # 3. Crear estructura final para el archivo .ini
    contenido = (
        "[RULES]\n"
        f"path = {path_str}\n"
        "variation = =1"
    )
    # 4. Guardar archivo
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write(contenido)   
    print(f"--- Archivo '{nombre_archivo}' generado ---")
    #print(contenido)
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

cantidad_reglas=100
cast_columns_int=["case:concept:name"]
cast_columns_date=["time:timestamp"]
ruta_csv="C:/Users/Diego/Documents/GitHub\Asistencia-graduada/Declarative_final/DeclarativeProcessSimulation/data/0.logs/PurchasingExample/"
ruta_log_filtrado_mejor='mi_proceso_filtrado_mejor.xes'
ruta_log_filtrado_peor='mi_proceso_filtrado_peor.xes'
ruta_modelo_declarativo_mejor="modelo_descubierto_mejor.csv"
ruta_modelo_declarativo_peor="modelo_descubierto_peor.csv"
pd.options.mode.chained_assignment = None


# carga del log de eventos cuando esta en formato xes
#log = pm4py.read_xes(ruta_xes, return_dataframe=True)
#df = pm4py.convert_to_dataframe(log)

# carga cuando es un archivo .csv el log de eventos
df = cargar_csvs_de_carpeta(ruta_csv)

# transformacion columnas a entero o date
convertir_int(cast_columns_int,df)
convertir_date(cast_columns_date,df)
df['lifecycle:transition'] = df['lifecycle:transition'].str.lower()

# --- CONFIGURACIÓN INICIAL ---
soporte_global = 0.2  # <-- IGUALADO AL SCRIPT SIMPLE
max_iteraciones = 200
iteracion = 0
reglas_recomendadas = set()
trazas_excluidas = [] 

df_analisis_trazas = generar_analisis_por_traza(df)
df_trabajo = df_analisis_trazas.copy() # Usar copy para evitar warnings

print("Iniciando Ciclo de Destilación Iterativa...")

# =====================================================================
# PASO 1: DEFINIR LA MEJOR TRAZA (FUERA DEL BUCLE - SE HACE UNA VEZ)
# =====================================================================
df_top_pca_global, _ = seleccion_optima_con_tabla(df_trabajo)
mejor_traza_id = df_top_pca_global['traza'].iloc[0]
print(f"La MEJOR TRAZA absoluta es: {mejor_traza_id}")

# Extraer log y modelo de la mejor traza
carga_df_to_format_xes(df[df['case:concept:name'] == mejor_traza_id], ruta_log_filtrado_mejor)
df_mejor = procesar_modelo_declarativo(ruta_log_filtrado_mejor, ruta_modelo_declarativo_mejor, soporte_global, "Mejor Caso")

# =====================================================================
# PASO 2: BUCLE DE DESTILACIÓN (Solo itera sobre las peores)
# =====================================================================
while (iteracion == 0 or len(reglas_recomendadas) > 10) and iteracion < max_iteraciones:
    iteracion += 1
    print(f"\n=== ITERACIÓN {iteracion} ===")
    
    # Filtrar trazas excluidas
    df_disponible = df_trabajo[~df_trabajo['traza'].isin(trazas_excluidas)].copy()
    
    if df_disponible.empty:
        print("Se agotaron las trazas disponibles.")
        break

    # Solo nos interesa la peor traza actual
    _, df_top_bti = seleccion_optima_con_tabla(df_disponible)
    peor_traza_actual = df_top_bti['traza'].iloc[0]
    trazas_excluidas.append(peor_traza_actual)
    
    #print(f"Podando con la Peor Traza actual: {peor_traza_actual}")

    # Generar XES y Modelo SOLO para la peor traza
    carga_df_to_format_xes(df[df['case:concept:name'] == peor_traza_actual], ruta_log_filtrado_peor)
    df_peor = procesar_modelo_declarativo(ruta_log_filtrado_peor, ruta_modelo_declarativo_peor, soporte_global, "Peor Caso")

    # Resta de Reglas
    if iteracion == 1:
        # En la iteración 1, es igual al código simple
        reglas_recomendadas = analizar_diferencia_reglas(df_mejor, df_peor)
    else:
        # En las siguientes iteraciones, podamos
        reglas_peor_actual = set(df_peor['Constraint'])
        reglas_recomendadas = reglas_recomendadas - reglas_peor_actual

    print(f"Reglas restantes tras podado: {len(reglas_recomendadas)}")
# aplicacion de filtro y descubrimiento de reglas a recomendar

# Seleccion rule recomendada y escritura archivo .ini con la notacion de declarative
if reglas_recomendadas:
    print("\nCantidad de reglas a promover para mejorar el proceso:",len(reglas_recomendadas))
    df_reglas = pd.DataFrame(list(reglas_recomendadas), columns=["Reglas_recomendadas"]).sort_values(by="Reglas_recomendadas").head(1)
    regla_a_recomendar =df_reglas["Reglas_recomendadas"].iloc[0]
    print("Regla seleccionada: ",regla_a_recomendar)
    convertir_declare_to_declarative(regla_a_recomendar,ruta_csv + "rules.ini")

