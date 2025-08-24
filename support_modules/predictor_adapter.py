import subprocess
import os

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from GenerativeLSTM.model_prediction import model_predictor as pr
import pandas as pd

# Useful functions

def get_latest_output_folder(output_path) -> str:
    # List all items in the output path
    folders = [os.path.join(output_path, f) for f in os.listdir(output_path)]
    # Filter to only include directories
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        return None
    # Sort by last modified time, descending
    latest_folder = max(folders, key=os.path.getmtime)
    return os.path.basename(latest_folder)



def run_simod_docker(input_path="data/1.input_logs", output_path="data/2.bps_asis", config_file_name="configuration_generated.yaml"):

    # Path to config inside the container (relative to mounted resources folder)
    config_inside_container = "/usr/src/Simod/resources/" + config_file_name
    docker_command = [
        "docker", "run", "--rm",
        "-v", f"{input_path}:/usr/src/Simod/resources",
        "-v", f"{output_path}:/usr/src/Simod/outputs",
        # "-w", "/usr/src",
        "nokal/simod",
        "poetry", "run", "simod",
        "--configuration", config_inside_container
    ]   


    print("Running Docker command:")
    print(" ".join(docker_command))
    result = subprocess.run(docker_command, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Simod ran successfully.\n")
        print(result.stdout)
    else:
        print("❌ Simod failed.\n")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError("Simod execution failed. Check the logs for details.")




def run_prosimos_docker(input_path="data/output_tobe", output_path="data/output_simulation", model_path="", resources_path=""):

    model_filename = os.path.basename(model_path)

    docker_command = [
        "docker", "run", "--rm",
        "-v", f"{input_path}:/usr/src/Simod/resources",
        "-v", f"{output_path}:/usr/src/Simod/outputs",
        # "-w", "/usr/src", 
        "simod_fixed",
        "poetry", "run", "prosimos",
        "start-simulation", "--bpmn_path", "/usr/src/Simod/resources/"+ model_path,
        "--json_path", "/usr/src/Simod/resources/" + resources_path,
        
        "--total_cases", "20",
        "--log_out_path", "/usr/src/Simod/outputs/" + model_filename.replace(".bpmn", "_promious_log.csv"),
        "--stat_out_path","/usr/src/Simod/outputs/" + model_filename.replace(".bpmn", "_promious_stats.csv"),
    ]

    print("Running Docker command:")
    print(" ".join(docker_command))
    pass
    result = subprocess.run(docker_command, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Promius ran successfully.\n")
        print(result.stdout)
    else:
        print("❌ Promius failed.\n")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError("Prosimos execution failed. Check the logs for details.")


# BIMP simulation

def run_bimp_docker(bimp_path, bpmn_path, csv_path):
    print(" -- Simulating Process with Dockerized Java --")

    # Prepare paths (make sure they use forward slashes)
    local_path = os.path.abspath(os.getcwd()).replace("\\", "/")
    print(f"Local path: {local_path}")
    # Docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{local_path}:/app",  # mount the local directory into Docker
        "-w", "/app",                # work inside /app
        "java8-xvfb",                 # the Docker image name
        "java", "-jar", bimp_path,   # run the simulator JAR
        bpmn_path,                 # input BPMN file
        "-csv", csv_path      # output CSV
    ]

    print("Running Docker Java Simulation...")
    print("Command:", " ".join(docker_cmd))

    try:
        result = subprocess.run(docker_cmd, stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("✅ Simulation was successfully executed")
            print(result.stdout)
        elif result.returncode == 1:
            lines = result.stdout.split('\n')
            exception_output = [lines[i - 1] for i in range(len(lines)) if 'BPSimulatorException' in lines[i]]
            print("❌ Execution failed:", ' '.join(exception_output))
        else:
            print(f"⚠️ Simulation failed with return code: {result.returncode}")
    except Exception as e:
        print(f"❌ Error running simulation: {str(e)}")



# ----- Resource Adapter (Adapts asis resources to generated JSON)


def extract_tasks(xml_tree):
    """
    Extracts task elements from BPMN XML.
    Returns a dict: {task_name: task_id}
    """
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    root = xml_tree.getroot()

    task_tags = [
        'task', 'userTask', 'serviceTask', 'scriptTask', 'manualTask',
        'businessRuleTask', 'receiveTask', 'sendTask', 'callActivity',
        'startEvent', 'endEvent'
    ]

    tasks = {}
    for tag in task_tags:
        for elem in root.findall(f'.//bpmn:{tag}', ns):
            name = elem.attrib.get('name')
            id_ = elem.attrib.get('id')
            if name:
                tasks[name] = id_
    return tasks

def join_tasks_by_name(file1_path, file2_path):
    """
    Joins two BPMN files by task name (case-insensitive).
    Returns dict: {original_case_task_name: (id_in_file1, id_in_file2)}
    """
    tree1 = ET.parse(file1_path)
    tree2 = ET.parse(file2_path)

    tasks1 = extract_tasks(tree1)
    tasks2 = extract_tasks(tree2)

    # Normalize task names to lowercase for matching
    tasks1_ci = {name.lower(): (name, tid) for name, tid in tasks1.items()}
    tasks2_ci = {name.lower(): (name, tid) for name, tid in tasks2.items()}

    all_normalized_names = set(tasks1_ci.keys()) | set(tasks2_ci.keys())
    result = {}
 
    for norm_name in all_normalized_names:
        orig_name1, id1 = tasks1_ci.get(norm_name, (None, None))
        orig_name2, id2 = tasks2_ci.get(norm_name, (None, None))

        # Prefer the original name from file1, or fallback to file2
        final_name = orig_name1 or orig_name2
        result[final_name] = (id1, id2)

    return result




def remap_ids_and_transfer_fields(original_json, generated_json, id_mapping):
    # Normalize id_mapping keys to lowercase for case-insensitive access
    normalized_mapping = {name.lower(): (ids[0], ids[1]) for name, ids in id_mapping.items()}
    # print("Normalized ID Mapping:")
    # for name, ids in normalized_mapping.items():
    #     print(f"{name}: {ids}")
    # Simple replacements
    for key in ["arrival_time_calendar", "arrival_time_distribution", "resource_calendars"]:
        generated_json[key] = original_json.get(key, [])

    # Remap assignedTasks in resource_profiles
    new_profiles = []
    for profile in original_json.get("resource_profiles", []):
        new_resource_list = []
        for res in profile.get("resource_list", []):
            new_tasks = []
            for old_id in res.get("assignedTasks", []):
                # Find name from original task_id
                name = next(
                    (task_name for task_name, (orig_id, _) in normalized_mapping.items()
                     if orig_id == old_id),
                    None
                )
                if name and normalized_mapping[name][1]:
                    new_tasks.append(normalized_mapping[name][1])
            res["assignedTasks"] = new_tasks
            new_resource_list.append(res)
        profile["resource_list"] = new_resource_list
        new_profiles.append(profile)
    generated_json["resource_profiles"] = new_profiles

    # Remap task_id in task_resource_distribution
    new_task_resource_distribution = []
    for item in original_json.get("task_resource_distribution", []):
        old_task_id = item["task_id"]
        name = next(
            (task_name for task_name, (orig_id, _) in normalized_mapping.items()
             if orig_id == old_task_id),
            None
        )
        if name and normalized_mapping[name][1]:
            item["task_id"] = normalized_mapping[name][1]
            new_task_resource_distribution.append(item)
    generated_json["task_resource_distribution"] = new_task_resource_distribution

    return generated_json


def adapt_json(
        asis_bpmn_path="",
        asis_json_path = "", 
        tobe_bpmn_path="",
        tobe_json_path = "",
        merged_json_path="merged.json",
        verbose=False
        
        ):

    # Load JSONs
    with open(asis_json_path) as f:
        original_json = json.load(f)

    with open(tobe_json_path) as f:
        generated_json = json.load(f)

    # Get BPMN ID mapping
    bpmn_mapping = join_tasks_by_name(asis_bpmn_path, tobe_bpmn_path)
    if verbose:
        print("BPMN IDs:")
        for name, ids in bpmn_mapping.items():
            print(f"{name}: {ids}")
    # Apply transformation
    updated_json = remap_ids_and_transfer_fields(original_json, generated_json, bpmn_mapping)

    # Save
    with open(merged_json_path, "w") as f:
        json.dump(updated_json, f, indent=4)


# Prediction 

def hallucinate(parameters, input_folder="",output_folder="", rules_path=""):
    #Generative model prediction
    # Crete output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(parameters)
    pr.ModelPredictor(parameters,input_folder=input_folder, output_folder=output_folder, rules_path=rules_path)
    # Remove unwanted tasks from the log
    log = pd.read_csv(os.path.join(output_folder, parameters['filename']))
    log = log[~log['task'].isin(['Start', 'End', 'start', 'end'])]
    log.to_csv(os.path.join(output_folder, parameters['filename']), index=False)

if __name__ == "__main__":
    run_prosimos_docker()
