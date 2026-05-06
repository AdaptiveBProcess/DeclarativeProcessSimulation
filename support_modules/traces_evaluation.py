import pandas as pd
import networkx as nx
import os
import itertools
import json
import configparser
from glob import glob

def get_stats_log_traces(traces_gen_path):
    
    traces_files = glob(os.path.join(traces_gen_path, '*.csv'))
    if len(traces_files)>0:
        df_traces = pd.read_csv(traces_files[0])
        if len(traces_files)>1:
            for traces_file in traces_files[1:]:
                df_trace_tmp = pd.read_csv(traces_file)
                df_traces = pd.concat([df_traces, df_trace_tmp])
    
        return df_traces, traces_files
    else:
        return [], []


def extract_rules(path='GenerativeLSTM/rules.ini',verbose=False):
    config = configparser.ConfigParser()
    config.read(path)

    settings = {}
    settings['path'] = [x.strip() for x in config['RULES']['path'].split('>>')]
    settings['variation'] = config['RULES']['variation'][0] if 'variation' in config.options('RULES') else None
    settings['prop_variation'] = float(config['RULES']['variation'][1:]) if 'variation' in config.options('RULES') else None
    if verbose:
        print ( "\n", "### Rules path: ", settings['path'] , "###" , "\n")

    
    if '*' in settings['path']:
        settings['rule'] = 'eventually'
    elif '^' in config['RULES']['path']:
        settings['rule'] = 'not_allowed'
    elif '>>' in config['RULES']['path'] and '*' not in settings['path'] and '^' not in settings['path'] and '|' not in settings['path'] and '#' not in settings['path'] and '+' not in settings['path'] and '-' not in settings['path']:
        settings['rule'] = 'directly'
    elif '>>' not in config['RULES']['path'] and '*' not in settings['path'] and '^' not in settings['path']:
        settings['rule'] = 'required'
    # nueva regla precedence    
    elif '>>' in config['RULES']['path'] and '|' in settings['path'] and '*' not in settings['path'] and '^' not in settings['path']: 
        settings['rule'] = 'precedence'
    # nueva regla succession
    elif '>>' in config['RULES']['path'] and '#' in settings['path'] and '*' not in settings['path'] and '^' not in settings['path']: 
        settings['rule'] = 'succession'
    # nueva regla coexistence
    elif '>>' in config['RULES']['path'] and '+' in settings['path'] and '*' not in settings['path'] and '^' not in settings['path']: #
        settings['rule'] = 'coexistence'
    # nueva regla NotChainSuccession
    elif '>>' in config['RULES']['path'] and '-' in settings['path'] and '*' not in settings['path'] and '^' not in settings['path']: #
        settings['rule'] = 'NotChainSuccession'

    settings['path'] = [x.replace('^', '') for x in settings['path'] if x != '*']
    #eliminar elemento "|","#","+","-"
    settings['path'] = [x for x in settings['path'] if x not in ['|', '#','+',"-"]]
    return settings

#list_case is the trace genererated by the allucinator
#ac_index is dictionary with the activity and its index
#act_paths is the activities flow
#rule
## evaluates if a trace comply with a rule
def evaluate_condition_list(list_case, ac_index, act_paths, rule):

    act_paths_idx = [ac_index[x] if x in ac_index.keys() else x for x in act_paths]
    u_tasks = [x for x in list(set(list_case))]
    
    #create graph from the allucinator trace
    G = nx.DiGraph()
    for task in u_tasks:
        G.add_node(task)

    order = [(a, b) for a, b in zip(list_case[:-1], list_case[1:])]
    G.add_edges_from(order)
    #end 

    #indicates if the trace comply with the rule
    if rule == 'eventually':
        act1 = act_paths_idx[0]
        act2 = act_paths_idx[1]

        # 1. Condición No-Vacua: act1 debe existir en el list_case
        if act1 in list_case:
            conds = True

            # 2. Encontrar TODOS los índices donde ocurre act1 en la traza
            act1_indices = [i for i, act in enumerate(list_case) if act == act1]

            # 3. Validar que para CADA ocurrencia de act1, act2 ocurra después
            for idx in act1_indices:
                # Revisamos el sub-arreglo que va DESPUÉS de act1
                if act2 not in list_case[idx + 1:]:
                    conds = False
                    break # Falla para una activación = falla toda la regla
        else:
            # Si act1 nunca se ejecutó, el cumplimiento es vacuo. Lo rechazamos.
            conds = False
    elif rule == 'not_allowed':
        conds = not(G.has_node(act_paths_idx[0]))
    elif rule == 'directly':
        if G.has_node(act_paths_idx[0]):  
            conds = G.has_edge(act_paths_idx[0],act_paths_idx[1])
        else:
            conds = False
    elif rule == 'required':
        conds = G.has_node(act_paths_idx[0])
    # Agregacion regla Precedence
    elif rule == 'precedence':
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])
        # La regla se activa por la presencia de B (el segundo nodo)
        if b_exists:
            # Si B existe, A DEBE existir y ser su ancestro
            conds = a_exists and (act_paths_idx[0] in nx.ancestors(G, act_paths_idx[1]))
        else:
            # Si B no existe, la regla no se "cumple" 
            conds = False
    # Agregacion regla Succession
    elif rule == 'succession':
        # 1. Verificamos existencia de ambos nodos
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])

        if a_exists and b_exists:
            # 2. Verificamos la direccionalidad
            path_a_b = nx.has_path(G, act_paths_idx[0], act_paths_idx[1])
            path_b_a = nx.has_path(G, act_paths_idx[1], act_paths_idx[0]) 
            # Se cumple SOLO si existe el camino A->B y NO el camino B->A (asimetría)
            conds = path_a_b and not path_b_a
        else:
            # 3. Si falta alguno (o ambos), la regla NO se cumple (evitamos vacuidad)
            conds = False
    # Agregacion regla coexistence
    elif rule == 'coexistence':
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])
        conds = a_exists and b_exists
    elif rule == 'NotChainSuccession':
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])
        # El disparador (trigger) es A
        if a_exists:
            # Si A existe, verificamos que NO tenga a B como sucesor directo
            is_direct_successor = b_exists and (act_paths_idx[1] in G.successors(act_paths_idx[0]))
            # Se cumple si NO es sucesor directo
            conds = not is_direct_successor
        else:
            # Si A ni siquiera ocurrió, la regla no se activó.
            # Bajo criterio de "no-vacuidad", esto es False.
            conds = False
    return conds

## receives a dataframe and orders
def evaluate_condition(df_case, ac_index, act_paths, rule):

    act_paths_idx = [ac_index[x] if x in ac_index.keys() else x for x in act_paths]
    df_case['rank'] = df_case.groupby('caseid')['start_timestamp'].rank().astype(int)
    df_case = df_case.sort_values(by='rank')
    u_tasks = [ac_index[x] for x in df_case['task'].drop_duplicates()]

    G = nx.DiGraph()
    for task in u_tasks:
        G.add_node(task)

    tasks = list(df_case['task'])
    if list(df_case['rank']) == list(set(list(df_case['rank']))):
        order = [(ac_index[x[0]], ac_index[x[1]]) for x in [(a, b) for a, b in zip(tasks[:-1], tasks[1:])]]
    else:
        order = []
        for i in range(1, len(df_case['rank'])):
            c_task = list(df_case[df_case['rank']==i]['task'])
            n_task = list(df_case[df_case['rank']==i+1]['task'])
            order += [(ac_index[x[0]], ac_index[x[1]]) for x in list(itertools.product(c_task, n_task))]

    G.add_edges_from(order)
    #indicates if the trace comply with the rule
    # 1. Creamos la secuencia de la traza usando los índices (para que coincida con act_paths_idx)
    # Usamos ac_index[x] para convertir el nombre de la tarea en su ID numérico
    trace_sequence = [ac_index[x] for x in tasks]

    if rule == 'eventually':
        act1 = act_paths_idx[0] # El ID de la actividad de activación
        act2 = act_paths_idx[1] # El ID de la actividad objetivo
        
        # Validación NO-VACUA
        if act1 in trace_sequence:
            conds = True
            # Buscamos todas las posiciones de la activación
            act1_indices = [i for i, val in enumerate(trace_sequence) if val == act1]
            
            for idx in act1_indices:
                # Si para alguna activación no existe el objetivo después, falla
                if act2 not in trace_sequence[idx + 1:]:
                    conds = False
                    break
        else:
            # Si la actividad 1 ni siquiera ocurre, es un cumplimiento vacuo (se rechaza)
            conds = False
    elif rule == 'not_allowed':
        conds = not(G.has_node(act_paths_idx[0]))
    elif rule == 'directly':
        if G.has_node(act_paths_idx[0]):  
            conds = G.has_edge(act_paths_idx[0],act_paths_idx[1])
        else:
            conds = False
    elif rule == 'required':
        conds = G.has_node(act_paths_idx[0])
    # Agregacion regla Precedence
    elif rule == 'precedence':
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])
        # La regla se activa por la presencia de B (el segundo nodo)
        if b_exists:
            # Si B existe, A DEBE existir y ser su ancestro
            conds = a_exists and (act_paths_idx[0] in nx.ancestors(G, act_paths_idx[1]))
        else:
            # Si B no existe, la regla no se "cumple" 
            conds = False
    # Agregacion regla Succession
    elif rule == 'succession':
        # 1. Verificamos existencia de ambos nodos
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])
        
        if a_exists and b_exists:
            # 2. Verificamos la direccionalidad
            path_a_b = nx.has_path(G, act_paths_idx[0], act_paths_idx[1])
            path_b_a = nx.has_path(G, act_paths_idx[1], act_paths_idx[0]) 
            
            # Se cumple SOLO si existe el camino A->B y NO el camino B->A (asimetría)
            conds = path_a_b and not path_b_a
        else:
            # 3. Si falta alguno (o ambos), la regla NO se cumple (evitamos vacuidad)
            conds = False 
    # Agregacion regla coexistence
    elif rule == 'coexistence':
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])
        conds = a_exists and b_exists  
    elif rule == 'NotChainSuccession':
        a_exists = G.has_node(act_paths_idx[0])
        b_exists = G.has_node(act_paths_idx[1])
        # El disparador (trigger) es A
        if a_exists:
            # Si A existe, verificamos que NO tenga a B como sucesor directo
            is_direct_successor = b_exists and (act_paths_idx[1] in G.successors(act_paths_idx[0]))
            # Se cumple si NO es sucesor directo
            conds = not is_direct_successor
        else:
            # Si A ni siquiera ocurrió, la regla no se activó.
            # Bajo criterio de "no-vacuidad", esto es False.
            conds = False
    return conds

class GenerateStats:
    def __init__(self, log, ac_index, act_paths, rule) -> None:
        self.log = log
        self.log['start_timestamp'] = pd.to_datetime(self.log['start_timestamp'])
        self.log['end_timestamp'] = pd.to_datetime(self.log['end_timestamp'])
        self.log['rank'] = self.log.groupby('caseid')['start_timestamp'].rank().astype(int)
        self.ac_index = ac_index
        self.act_paths = act_paths
        self.rule = rule

    def get_stats(self):

        pos_cases = 0
        total_cases = len(self.log['caseid'].drop_duplicates())
        for caseid in self.log['caseid'].drop_duplicates():

            df_case = self.log[self.log['caseid']==caseid]
            cond = evaluate_condition(df_case, self.ac_index, self.act_paths, self.rule)
            if cond:
                pos_cases += 1

        return pos_cases, total_cases