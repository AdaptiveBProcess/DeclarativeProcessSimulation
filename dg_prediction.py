import support_modules.predictor_adapter as pa
import support_modules.bimp_parser as bp
import os
import getopt
import sys
import gzip
import shutil
import pandas as pd

from support_modules import traces_evaluation as te



def generate_bps_model(input_folder="log", output_folder="bps", config_file_name="configuration.yaml"):
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    # Crete output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    pa.run_simod_docker(input_path=input_folder, output_path=output_folder, config_file_name=config_file_name)

def simulate_model(input_path, output_path, bpmn_filename, resources_filename):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    # Crete output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    bpmn_path = pa.get_latest_output_folder(input_path) + "/best_result/" + bpmn_filename
    resources_path = pa.get_latest_output_folder(input_path) + "/best_result/" + resources_filename
    pa.run_prosimos_docker(input_path=input_path, output_path=output_path, model_path=bpmn_path, resources_path=resources_path)

def adapt_resources(original_folder, original_filename, generated_folder, generated_filename, merged_filename):
    original_folder = os.path.join(original_folder,pa.get_latest_output_folder(original_folder) + "/best_result/")
    generated_folder = os.path.join(generated_folder,pa.get_latest_output_folder(generated_folder) + "/best_result/")
    pa.adapt_json(
        asis_bpmn_path= os.path.join(original_folder,original_filename),
        asis_json_path= os.path.join(original_folder,original_filename.replace(".bpmn", ".json")),
        tobe_bpmn_path= os.path.join(generated_folder, generated_filename),
        tobe_json_path=  os.path.join(generated_folder, generated_filename.replace(".bpmn", ".json")),
        merged_json_path= os.path.join(generated_folder, merged_filename)   
    )


def generate_params(argv, filename="", path=""):
    def map_opt(opt):
        return {'-h': 'help', '-a': 'activity', '-c': 'folder',
                '-b': 'model_file', '-v': 'variant', '-r': 'rep'}.get(opt)

    params = {
        'one_timestamp': False,
        'include_org_log': False,
        'read_options': {
            # 'timeformat': "%Y-%m-%d %H:%M:%S%z",
            'timeformat': "%Y-%m-%d %H:%M:%S",
            'column_names': {
                'Case ID': 'caseid',
                'Activity': 'task',
                'lifecycle:transition': 'event_type',
                'Resource': 'user'
            },
            'one_timestamp': False,
            'filter_d_attrib': False
        },
        'filename': filename,
        # 'input_path': 'GenerativeLSTM/input_files',
        # 'sm3_path': os.path.join('GenerativeLSTM', 'external_tools', 'splitminer3', 'bpmtk.jar'),
        # 'bimp_path': os.path.join('GenerativeLSTM', 'external_tools', 'bimp', 'qbp-simulator-engine_with_csv_statistics.jar'),
        # 'concurrency': 0.0,
        # 'epsilon': 0.5,
        # 'eta': 0.7
    }

    if not argv:
        name = params['filename'].split('.')[0]
        params.update({
            'activity': 'pred_log',
            'folder': pa.get_latest_output_folder(path),
            'model_file': name + '.h5',
            'log_name': name,
            'is_single_exec': False,
            'variant': 'Rules Based Random Choice',
            'rep': 1
        })
    else:
        try:
            opts, _ = getopt.getopt(argv, "ho:a:f:c:b:v:r:")
            for opt, arg in opts:
                key = map_opt(opt)
                if key:
                    params[key] = int(arg) if key == 'rep' else arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)

    return params



def call_predict(parameters, input_folder="",output_folder="", rules_path=""):
    pa.hallucinate(
        parameters=parameters,
        input_folder=input_folder,
        output_folder=output_folder,
        rules_path=rules_path
    )

def compress_csv_to_gz(csv_file_path, output_folder=None):
    """
    Compress a .csv file to .csv.gz format.

    Parameters:
        csv_file_path (str): Full path to the input CSV file.
        output_folder (str, optional): Directory where the .gz file should be saved.
                                       If None, saves in the same directory as the input file.
    """
    if not os.path.isfile(csv_file_path) or not csv_file_path.lower().endswith('.csv'):
        raise ValueError("Provided file must be a valid .csv file.")

    base_name = os.path.basename(csv_file_path)
    gz_file_name = base_name + '.gz'

    if output_folder is None:
        output_folder = os.path.dirname(csv_file_path)

    os.makedirs(output_folder, exist_ok=True)
    gz_file_path = os.path.join(output_folder, gz_file_name)

    with open(csv_file_path, 'rb') as f_in, gzip.open(gz_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Compressed file saved to: {gz_file_path}")
    return gz_file_path


def extract_rules(path):
    rules = te.extract_rules(path=path)
    rules = f"{rules['rule']}__"+"__".join(item.replace(' ', '_') for item in rules['path'])
    return rules

def simulate_bimp(input_path="", output_path="", NAME="", bimp_path="./GenerativeLSTM/external_tools/bimp/qbp-simulator-engine.jar"):
    
    final_input_path = f"{input_path}/{pa.get_latest_output_folder(input_path)}/best_result"
    bpmn_bimp_path = f"{final_input_path}/{NAME}_bimp_version.bpmn"

    bp.embed_qbp_simulation(
        bpmn_path=f"{final_input_path}/{NAME}.bpmn",
        resources_json_path=f"{final_input_path}/{NAME}_merged.json",
        bpmn_bimp_path=bpmn_bimp_path)
    pa.run_bimp_docker(
        bimp_path=bimp_path,
        bpmn_path=bpmn_bimp_path,
        csv_path=f"{output_path}/{NAME}_bimp_log.csv"
    )


def main(argv):

    FILENAME = "PurchasingExample.csv"
    NAME = FILENAME.split('.')[0]
    merged_filename = FILENAME.replace(".csv", "_merged.json")

    rules_path = f"data/0.logs/{NAME}/rules.ini"
    rules_name = extract_rules(rules_path)

    path_prediction_models = f"data/1.predicton_models/{NAME}"

    call_predict(generate_params(argv,filename=FILENAME, path=path_prediction_models), input_folder=path_prediction_models, output_folder=f"data/2.hallucination_logs/{NAME}", rules_path=rules_path)

    compress_csv_to_gz(f"data/0.logs/{NAME}/{FILENAME}",output_folder=f"data/2.input_logs/{NAME}")
    compress_csv_to_gz(f"data/2.hallucination_logs/{NAME}/{FILENAME}")

    # Generate the asis model
    # print("Generating the asis model...")
    generate_bps_model(input_folder=f"data/2.input_logs/{NAME}", output_folder=f"data/3.bps_asis/{NAME}", config_file_name=f"configuration_original.yaml")

    print("Generating the tobe model...")
    # Generate the tobe model
    generate_bps_model(input_folder=f"data/2.hallucination_logs/{NAME}", output_folder=f"data/3.bps_tobe/{NAME}", config_file_name=f"configuration_generated.yaml")


    # Merge resources
    adapt_resources(original_folder=f"data/3.bps_asis/{NAME}", original_filename=f"{FILENAME.replace('.csv', '.bpmn')}", 
                    generated_folder=f"data/3.bps_tobe/{NAME}", generated_filename=f"{FILENAME.replace('.csv', '.bpmn')}",
                    merged_filename= merged_filename)
    

    # Simulate the model
    simulate_model(input_path=f"data/3.bps_tobe/{NAME}", output_path=f"data/4.simulation_results/{NAME}/{rules_name}", bpmn_filename=f"{FILENAME.replace('.csv', '.bpmn')}",
                   resources_filename= merged_filename)

    simulate_bimp(input_path=f"data/3.bps_tobe/{NAME}", output_path=f"data/4.simulation_results/{NAME}/{rules_name}", NAME=NAME)

if __name__ == "__main__":
    main(sys.argv[1:])