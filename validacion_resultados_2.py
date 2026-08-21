import pandas as pd
import io
import numpy as np

# Configuración de visualización de Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def analizar_metricas_archivo(file_path):
    """Lee un archivo de prosimos_stats y retorna el DataFrame de métricas."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    start_index = -1
    for i, line in enumerate(lines):
        if "Overall Scenario Statistics" in line:
            start_index = i + 1
            break
            
    if start_index == -1:
        raise ValueError(f"No se encontró 'Overall Scenario Statistics' en {file_path}")

    csv_content = "".join([l for l in lines[start_index:] if l.strip()])
    column_names = ["KPI", "Min", "Max", "Average", "Accumulated Value", "Trace Ocurrences"]
    
    df = pd.read_csv(io.StringIO(csv_content), names=column_names, header=0)
    df['KPI'] = df['KPI'].str.strip()
    df.set_index('KPI', inplace=True)
    return df

def promediar_iteraciones(rutas_archivos):
    """Itera sobre archivos y calcula promedios y desviaciones estándar."""
    if not rutas_archivos:
        return None
        
    resultados = []
    for i, ruta in enumerate(rutas_archivos):
        df = analizar_metricas_archivo(ruta)
        try:
            metrics = {
                'cycle_time': df.loc['cycle_time', 'Average'],
                'processing_time': df.loc['processing_time', 'Average'],
                'waiting_time': df.loc['waiting_time', 'Average']
            }
            # Opcional: incluir idle_time si existe
            metrics['idle_time'] = df.loc['idle_time', 'Average'] if 'idle_time' in df.index else 0
            
            # Cálculos de eficiencia
            metrics['PCE (%)'] = (metrics['processing_time'] / metrics['cycle_time']) * 100
            metrics['IL (%)'] = (metrics['waiting_time'] / metrics['cycle_time']) * 100
            resultados.append(metrics)
        except KeyError as e:
            print(f"Error extrayendo métricas en {ruta}: {e}")
            
    df_resultados = pd.DataFrame(resultados)
    df_promedios = df_resultados.mean().to_frame(name='Promedio')
    df_promedios['Desv. Estandar'] = df_resultados.std()
    
    return df_promedios

def formatear_resumen(df_promedios):
    """Convierte los resultados al formato científico: 'Media ± Desv' """
    if df_promedios is None:
        return {}
        
    resultados_str = {}
    for metrica in df_promedios.index:
        media = df_promedios.loc[metrica, 'Promedio']
        desv = df_promedios.loc[metrica, 'Desv. Estandar']
        
        # Si la desviación es NaN (ej. solo corriste 1 iteración), se pone 0.00
        if pd.isna(desv): desv = 0.0
        
        # Formatear con % si es PCE o IL
        if metrica in ['PCE (%)', 'IL (%)']:
            resultados_str[metrica] = f"{media:.2f}% ± {desv:.2f}%"
        else:
            # Formato estándar para tiempos (puedes ajustar los decimales si quieres)
            resultados_str[metrica] = f"{media:.2f} ± {desv:.2f}"
            
    return resultados_str

# =====================================================================
# 1. DEFINICIÓN DE RUTAS (Rellena esto conforme tengas los resultados)
# =====================================================================

# --- LOG: BPI CHALLENGE 2012 ---
bpic_asis = [
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\asis_20260518_142755\BPI_Challenge_2012_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\asis_20260518_164332\BPI_Challenge_2012_prosimos_stats.csv"
]
bpic_tobe_rec = [
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\coexistence__A_DECLINED__W_Afhandelen_leads_20260518_163838\BPI_Challenge_2012_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\coexistence__A_DECLINED__W_Afhandelen_leads_20260518_163839\BPI_Challenge_2012_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\coexistence__A_DECLINED__W_Afhandelen_leads_20260518_142755\BPI_Challenge_2012_prosimos_stats.csv"
]
bpic_tobe_global = [
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\NotChainSuccession__W_Completeren_aanvraag__A_PARTLYSUBMITTED_20260518_164332\BPI_Challenge_2012_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\NotChainSuccession__W_Completeren_aanvraag__A_PARTLYSUBMITTED_20260518_191341\BPI_Challenge_2012_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\BPI_Challenge_2012\NotChainSuccession__W_Completeren_aanvraag__A_PARTLYSUBMITTED_20260518_191342\BPI_Challenge_2012_prosimos_stats.csv"
] 


# --- LOG: RUNNING EXAMPLE ---
re_asis = [
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\asis_20260516_183130\RunningExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\asis_20260517_162559\RunningExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\asis_20260514_181222\RunningExample_prosimos_stats.csv"
]
re_tobe_rec = [
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\coexistence__EVENT_1_START__Task_D_20260514_181222\RunningExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\coexistence__EVENT_1_START__Task_D_20260516_183130\RunningExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\coexistence__EVENT_1_START__Task_D_20260516_184836\RunningExample_prosimos_stats.csv"
] 
re_tobe_global = [

    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\coexistence__EVENT_1_START__EVENT_7_END_20260517_162559\RunningExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\coexistence__EVENT_1_START__EVENT_7_END_20260517_164642\RunningExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\coexistence__EVENT_1_START__EVENT_7_END_20260517_164643\RunningExample_prosimos_stats.csv"
]


# --- LOG: PURCHASING EXAMPLE ---
pe_asis = [
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\asis_20260514_222134\PurchasingExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\asis_20260516_165820\PurchasingExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\asis_20260517_183212\PurchasingExample_prosimos_stats.csv"
] 
pe_tobe_rec = [
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\coexistence__\PurchasingExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\coexistence__1\PurchasingExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\coexistence__2\PurchasingExample_prosimos_stats.csv"
]
pe_tobe_global = [
   r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\NotChainSuccession__\PurchasingExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\NotChainSuccession__1\PurchasingExample_prosimos_stats.csv",
    r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\PurchasingExample\NotChainSuccession__2\PurchasingExample_prosimos_stats.csv"
]

# =====================================================================
# 2. PROCESAMIENTO CENTRALIZADO
# =====================================================================

# Diccionario maestro para almacenar todos los cálculos
data_maestra = {
    "Running Example": {
        "AS-IS": formatear_resumen(promediar_iteraciones(re_asis)),
        "TO-BE Top-Support": formatear_resumen(promediar_iteraciones(re_tobe_global)),
        "TO-BE GENESIS (Rec)": formatear_resumen(promediar_iteraciones(re_tobe_rec))
    },
    "Purchasing Example": {
        "AS-IS": formatear_resumen(promediar_iteraciones(pe_asis)),
        "TO-BE Top-Support": formatear_resumen(promediar_iteraciones(pe_tobe_global)),
        "TO-BE GENESIS (Rec)": formatear_resumen(promediar_iteraciones(pe_tobe_rec))
    },
    "BPI Challenge 2012": {
        "AS-IS": formatear_resumen(promediar_iteraciones(bpic_asis)),
        "TO-BE Top-Support": formatear_resumen(promediar_iteraciones(bpic_tobe_global)),
        "TO-BE GENESIS (Rec)": formatear_resumen(promediar_iteraciones(bpic_tobe_rec))
    }
}

# =====================================================================
# 3. CONSTRUCCIÓN DE LA TABLA FINAL
# =====================================================================

filas_tabla = []
escenarios = ["AS-IS", "TO-BE Top-Support", "TO-BE GENESIS (Rec)"]
metricas = ['PCE (%)', 'IL (%)', 'cycle_time', 'processing_time', 'waiting_time']

# AQUÍ SE HIZO EL CAMBIO: Primero iteramos por Métrica y luego por Escenario
for metrica in metricas:
    for escenario in escenarios:
        # Fila base (puse la métrica primero para que visualmente se agrupe mejor)
        fila = {"Métrica": metrica, "Escenario": escenario}
        
        # Buscar el valor en cada Log
        for log_name in data_maestra.keys():
            try:
                # Si el dato existe, lo extrae; si no, pone N/A
                valor = data_maestra[log_name][escenario].get(metrica, "N/A")
            except AttributeError:
                valor = "N/A"
            fila[log_name] = valor
            
        filas_tabla.append(fila)

df_tabla_final = pd.DataFrame(filas_tabla)

print("\n" + "="*80)
print(" TABLA MAESTRA DE RESULTADOS DE EVALUACIÓN")
print("="*80)
print(df_tabla_final.to_string(index=False))


#Descomenta la siguiente línea cuando quieras el código en LaTeX listo para copiar y pegar:

print("\n" + "="*80 + "\n CÓDIGO LATEX \n" + "="*80)
print(df_tabla_final.to_latex(index=False))
