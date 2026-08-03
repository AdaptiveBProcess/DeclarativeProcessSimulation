# %%
import pandas as pd
import numpy as np
import pm4py
import subprocess
import json
import os
import logging
import re
from scipy.stats import wasserstein_distance
from collections import Counter


ruta_gen=r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\2.hallucination_logs\RunningExample\RunningExample.csv"
ruta_real=r"C:\Users\Diego\Documents\GitHub\Asistencia-graduada\Declarative_final\DeclarativeProcessSimulation\data\4.simulation_results\RunningExample\metricas\test_split.csv"
df_gen=pd.read_csv(ruta_gen)
df_real=pd.read_csv(ruta_real)

df_real['start_timestamp'] = pd.to_datetime(df_real['start_timestamp'])
df_real['end_timestamp'] = pd.to_datetime(df_real['end_timestamp'])

df_gen['start_timestamp'] = pd.to_datetime(df_gen['start_timestamp'])
df_gen['end_timestamp'] = pd.to_datetime(df_gen['end_timestamp'])

df_real.rename(columns={'start_timestamp': 'timestamp',"caseid":"case_id","task":"activity"}, inplace=True)
df_gen.rename(columns={'start_timestamp': 'timestamp' ,"caseid":"case_id","task":"activity"}, inplace=True)

# print(df_gen.head())
# df_gen.dtypes
# print(df_real.head())
#df_real.dtypes

# %%
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from collections import Counter

class EvaluadorLogs:
    def __init__(self, df_test, df_gen):
        """
        Inicializa el evaluador asegurando el formato correcto de las columnas.
        """
        self.df_test = self._preparar_datos(df_test.copy())
        self.df_gen = self._preparar_datos(df_gen.copy())

    def _preparar_datos(self, df):
        # Aseguramos que el timestamp sea un objeto datetime para poder operar matemáticamente
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ordenamos cronológicamente por traza, vital para el flujo de control
        return df.sort_values(by=['case_id', 'timestamp']).reset_index(drop=True)

    def calcular_red(self):
        """
        RED (Relative Event Distribution): Mide el porcentaje de avance en el que ocurre cada evento.
        """
        def extraer_posiciones_relativas(df):
            # Encontramos el tiempo de inicio y fin de cada traza
            min_t = df.groupby('case_id')['timestamp'].transform('min')
            max_t = df.groupby('case_id')['timestamp'].transform('max')
            
            # Duración total de la traza en segundos
            duracion = (max_t - min_t).dt.total_seconds()
            
            # Calculamos la posición relativa (evitando división por cero)
            pos_relativa = np.zeros(len(df))
            mask = duracion > 0
            
            # (Tiempo actual - Tiempo inicial) / Duración total
            pos_relativa[mask] = (df.loc[mask, 'timestamp'] - min_t[mask]).dt.total_seconds() / duracion[mask]
            return pos_relativa

        red_test = extraer_posiciones_relativas(self.df_test)
        red_gen = extraer_posiciones_relativas(self.df_gen)
        
        # Distancia EMD entre ambas "montañas" de porcentajes
        return wasserstein_distance(red_test, red_gen)

    def calcular_ctd(self):
        """
        CTD (Cycle Time Distribution): Mide la duración total de las trazas.
        """
        def extraer_tiempos_de_ciclo(df):
            # Agrupamos por traza y calculamos la diferencia entre el último y el primer evento
            tiempos = df.groupby('case_id')['timestamp'].max() - df.groupby('case_id')['timestamp'].min()
            # Retornamos la duración en horas (puedes cambiarlo a días dividiendo por 24)
            return tiempos.dt.total_seconds() / 3600.0

        ctd_test = extraer_tiempos_de_ciclo(self.df_test)
        ctd_gen = extraer_tiempos_de_ciclo(self.df_gen)
        
        # Distancia EMD entre las distribuciones de tiempos totales
        return wasserstein_distance(ctd_test, ctd_gen)

    def calcular_2gd(self):
        """
        2GD (2-Gram Distance): Evalúa las frecuencias de las transiciones directas (A -> B).
        """
        def extraer_probabilidades_2gramas(df):
            # Desplazamos la columna actividad para tener el "siguiente evento" en la misma fila
            df['next_activity'] = df.groupby('case_id')['activity'].shift(-1)
            
            # Eliminamos el último evento de cada traza (porque no tiene un "siguiente")
            df_pares = df.dropna(subset=['next_activity'])
            
            # Creamos los pares y contamos su frecuencia
            pares = list(zip(df_pares['activity'], df_pares['next_activity']))
            conteos = Counter(pares)
            total = sum(conteos.values())
            
            # Convertimos conteos a probabilidades (porcentajes)
            return {par: conteo / total for par, conteo in conteos.items()}

        probs_test = extraer_probabilidades_2gramas(self.df_test)
        probs_gen = extraer_probabilidades_2gramas(self.df_gen)
        
        # Unificamos el vocabulario de 2-gramas (todos los pares que existan en ambos mundos)
        todos_los_pares = set(probs_test.keys()).union(set(probs_gen.keys()))
        
        # Construimos los vectores alineados
        vector_test = [probs_test.get(par, 0.0) for par in todos_los_pares]
        vector_gen = [probs_gen.get(par, 0.0) for par in todos_los_pares]
        
        # Distancia EMD entre los vectores de frecuencia
        return wasserstein_distance(vector_test, vector_gen)


#Instanciamos la clase con tus DataFrames
evaluador = EvaluadorLogs(df_real, df_gen)

print(f"Distancia RED: {evaluador.calcular_red():.4f}")
print(f"Distancia CTD: {evaluador.calcular_ctd():.4f}")
print(f"Distancia 2GD: {evaluador.calcular_2gd():.4f}")

