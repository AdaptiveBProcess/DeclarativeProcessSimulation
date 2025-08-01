# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:27:58 2025

@author: David Sequera
"""
import os
import sys
import getopt
from support_modules.predictor_adapter import get_latest_output_folder
# from get_folder import ReturnFolderName
from GenerativeLSTM.model_training import model_trainer as tr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



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
    FILENAME = 'Production.csv'
    NAME = FILENAME.split('.')[0]
    args = parse_args(argv,filename=FILENAME)

    parameters = {
        'read_options': {
            # 'timeformat': '%Y-%m-%dT%H:%M:%S.%fZ',
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            # 'timeformat': 'mixed',
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

    # Train the model
    os.makedirs(f'data/1.predicton_models/{NAME}', exist_ok=True)
    tr.ModelTrainer(parameters, input_folder=f'data/0.logs/{NAME}', output_folder=f'data/1.predicton_models/{NAME}')
    print(get_latest_output_folder("data/1.predicton_models"))


if __name__ == "__main__":
    main(sys.argv[1:])
