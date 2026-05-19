import pandas as pd
import io
pd.set_option('display.float_format', lambda x: '%.2f' % x)
def analizar_metricas_tobe(file_path):
    # 1. Leer el archivo buscando la sección específica
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    start_index = -1
    for i, line in enumerate(lines):
        if "Overall Scenario Statistics" in line:
            # La tabla empieza justo después del encabezado de la sección
            start_index = i + 1
            break
            
    if start_index == -1:
        return "No se encontró la sección 'Overall Scenario Statistics'."

    # 2. Leer la tabla. Usamos las columnas solicitadas.
    # Nota: El CSV de Prosimos a veces trae una línea vacía después del título, 
    # por eso verificamos que la línea tenga contenido.
    csv_content = "".join([l for l in lines[start_index:] if l.strip()])
    
    # Definimos los nombres de las columnas según tu requerimiento
    column_names = ["KPI", "Min", "Max", "Average", "Accumulated Value", "Trace Ocurrences"]
    
    df = pd.read_csv(io.StringIO(csv_content), names=column_names, header=0)

    # 3. Limpieza: Asegurarnos de que el KPI sea el índice para fácil acceso
    df['KPI'] = df['KPI'].str.strip()
    df.set_index('KPI', inplace=True)

    # 4. Cálculo del PCE (Process Cycle Efficiency)
    # PCE = (Processing Time / Cycle Time) * 100
    try:
        avg_processing = df.loc['processing_time', 'Average']
        avg_waiting = df.loc['waiting_time', 'Average']
        avg_cycle = df.loc['cycle_time', 'Average']
        pce = (avg_processing / avg_cycle) * 100
        il = (avg_waiting / avg_cycle) * 100
    except KeyError:
        pce = 0

    return df, pce,il


# --- Ejecución ---
file_name_asis = r"C:\Users\Diego\Desktop\PRUEBAS-RECOMENDACION-ASISTENCIA\archivos_primera_implementacion\RunningExample\salida_asis\RunningExample_prosimos_stats.csv"
#file_name_asis=r"C:\Users\Diego\Desktop\PRUEBAS-RECOMENDACION-ASISTENCIA\archivos_primera_implementacion\PurchasingExample\salida_asis\PurchasingExample_prosimos_stats.csv"
#file_name_asis=r"C:\Users\Diego\Desktop\PRUEBAS-RECOMENDACION-ASISTENCIA\archivos_primera_implementacion\BPI Challenge 2012\salida-asis\BPI_Challenge_2012_prosimos_stats.csv"

file_name_tobe = r"C:\Users\Diego\Desktop\PRUEBAS-RECOMENDACION-ASISTENCIA\archivos_primera_implementacion\RunningExample\salida_to_be\RunningExample_prosimos_stats.csv"
#file_name_tobe=r"C:\Users\Diego\Desktop\PRUEBAS-RECOMENDACION-ASISTENCIA\archivos_primera_implementacion\PurchasingExample\salida_tobe\PurchasingExample_prosimos_stats.csv"
#file_name_tobe=r"C:\Users\Diego\Desktop\PRUEBAS-RECOMENDACION-ASISTENCIA\archivos_primera_implementacion\BPI Challenge 2012\salida-tobe\BPI_Challenge_2012_prosimos_stats.csv"

df_final_asis, pce_calculado_asis, il_asis = analizar_metricas_tobe(file_name_asis)
df_final_tobe, pce_calculado_tobe, il_tobe = analizar_metricas_tobe(file_name_tobe)

print("=== REPORTE DE MÉTRICAS ASIS  ===")
print(df_final_asis.to_string())
print("-" * 30)
print(f"PROCESS CYCLE EFFICIENCY (PCE): {pce_calculado_asis:.2f}%")
print(f"LATENCY INDEX (LI): {il_asis:.2f}%")


print("=== REPORTE DE MÉTRICAS TOBE  ===")
print(df_final_tobe.to_string())
print("-" * 30)
print(f"PROCESS CYCLE EFFICIENCY (PCE): {pce_calculado_tobe:.2f}%")
print(f"LATENCY INDEX (LI): {il_tobe:.2f}%")
# %%