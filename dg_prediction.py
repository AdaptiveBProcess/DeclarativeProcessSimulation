import support_modules.predictor_adapter as pa
import support_modules.bimp_parser as bp
import os
import getopt
import sys
import gzip
import shutil
import pandas as pd
from params.Params import Params
from pathlib import Path
from typing import Optional, Union

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



def compress_csv_to_gz(csv_file_path: Path, output_folder: Optional[Path] = None):

    """
    Compress a .csv file to .csv.gz format.
 
    Parameters:
        csv_file_path (str): Full path to the input CSV file.
        output_folder (str, optional): Directory where the .gz file should be saved.
                                       If None, saves in the same directory as the input file.
    """
    csv_file_path = Path(csv_file_path)

    if not csv_file_path.is_file() or csv_file_path.suffix.lower() != ".csv":
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

def simulate_bimp(input_path="", output_path="", NAME="", PATH="", bimp_path="./GenerativeLSTM/external_tools/bimp/qbp-simulator-engine.jar"):
    
    final_input_path = f"{input_path}/{pa.get_latest_output_folder(input_path)}/best_result"
    bpmn_bimp_path = f"{final_input_path}/{NAME}_bimp_version.bpmn"

    bp.embed_qbp_simulation(
        bpmn_path=f"{final_input_path}/{NAME}.bpmn",
        resources_json_path=f"{final_input_path}/{NAME}_merged.json",
        bpmn_bimp_path=bpmn_bimp_path,
        exclusive=True
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
        log_filename="PurchasingExample.csv",
    )

    r = params.routes
    sim = params.simulation

    rules_name = extract_rules(r["rules"])
    pretty_print_params(r, title="Routes Paths")
    pretty_print_params(sim, title="Simulation Parameters")

    call_predict(
        sim,
        input_folder=r["models"],
        output_folder=r["hallucinated"],
        rules_path=r["rules"],
        root_path=params.root,
    )

    compress_csv_to_gz(r["log"] / params.log_filename, output_folder=r["input"])
    compress_csv_to_gz(r["hallucinated"] / params.log_filename)

    generate_bps_model(r["input"], r["bps_asis"], "configuration_original.yaml")
    generate_bps_model(r["hallucinated"], r["bps_tobe"], "configuration_generated.yaml")

    adapt_resources(
        original_folder=r["bps_asis"],
        original_filename=r["bpmn"],
        generated_folder=r["bps_tobe"],
        generated_filename=r["bpmn"],
        merged_filename=r["merged"],
    )

    simulate_model(
        input_path=r["bps_tobe"],
        output_path=r["simulation"] / rules_name,
        bpmn_filename=r["bpmn"],
        resources_filename=r["merged"],
    )

    simulate_bimp(
        input_path=r["bps_tobe"],
        output_path=r["simulation"] / rules_name,
        NAME=params.name,
        PATH=params.root,
    )

if __name__ == "__main__":
    main(sys.argv[1:])