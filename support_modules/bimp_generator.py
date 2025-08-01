import os
import subprocess

def sm3_miner_docker(xes_file, output_path=None):
    print(" -- Mining Process Structure with Dockerized Java 8 + Xvfb --")

    # Classpath for Java inside the container (Linux-style paths)
    classpath = (
        "GenerativeLSTM/external_tools/splitminer3/bpmtk.jar:"
        "GenerativeLSTM/external_tools/splitminer3/lib/*"
    )

    # Host path to mount (make sure it's absolute and Docker-friendly)
    local_path = os.path.abspath(os.getcwd()).replace("\\", "/")

    # Java command to run inside the container
    java_cmd = (
        f'Xvfb :99 -screen 0 1024x768x16 & '
        f'java -cp "{classpath}" '
        f'au.edu.unimelb.services.ServiceProvider '
        f'SMD {0.5} {0.7} false false false '
        f'{xes_file} {output_path}'
    )

    # Final Docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{local_path}:/app",
        "-w", "/app",
        "java8-xvfb",
        "sh", "-c", java_cmd
    ]

    print("Running Docker Java 8 SplitMiner3 with Xvfb...")
    print("Command:", " ".join(docker_cmd))

    # Check input file exists
    host_xes_path = os.path.join(os.getcwd(), xes_file)
    if not os.path.exists(host_xes_path):
        print(f"❌ Error: Input file not found: {host_xes_path}")
        os.makedirs(os.path.dirname(host_xes_path), exist_ok=True)
        print(f"Please ensure the file exists at {host_xes_path}")
        return False

    # Optional debugging pause
    # input("Press Enter to continue (debugging pause)...")

    try:
        result = subprocess.run(docker_cmd, check=False)

        if result.returncode != 0:
            print(f"❌ SplitMiner3 execution failed with return code {result.returncode}")
            return False
        else:
            print("✅ SplitMiner3 finished successfully.")
            return True
    except Exception as e:
        print(f"❌ Error executing SplitMiner3: {str(e)}")
        return False
    

if __name__ == "__main__":
    # Example usage
    xes_file = "./data/4.simulation_results/PurchasingExample/not_allowed__Amend_Request_for_Quotation\PurchasingExample_promious_log_not_amend.csv"
    output_path = "./data/4.simulation_results/PurchasingExample/not_allowed__Amend_Request_for_Quotation\PurchasingExample_promious_log_not_amend2.bpmn"
    sm3_miner_docker(xes_file, output_path)