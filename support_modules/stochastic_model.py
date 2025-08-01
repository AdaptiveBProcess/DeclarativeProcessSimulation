import warnings
warnings.filterwarnings('ignore')

from support_modules.log_replayer_stochastic import LogReplayerS
from readers.process_structure import create_process_structure
import readers.log_reader as lr
import readers.bpmn_reader as br

from bs4 import BeautifulSoup
import platform as pl
import subprocess
import os

class StochasticModel:
    def __init__(self, settings):
        self.settings = settings
        self.load_structures()
        self.extract_stochastic_model()
        
    def load_structures(self):
        #self.log_path = os.path.join('GenerativeLSTM','input_files', 'spmd', 'ConsultaDataMining201618.csv')
        self.log_path = os.path.join('GenerativeLSTM','input_files', 'spmd', self.settings['file'] + '.csv')
        self.log = lr.LogReader(self.log_path, self.settings)
        self._sm3_miner_docker()
        self.bpmn = br.BpmnReader(self.settings['tobe_bpmn_path'])
        self.model = create_process_structure(self.bpmn)

    def extract_stochastic_model(self):
        self.lrs = LogReplayerS(self.settings['tobe_bpmn_path'], self.bpmn, self.model, self.log)

    def change_branch_node(self, key_node, key_task):
        for seq_flow in self.lrs.sequence_flows:
            if seq_flow.get('id') == key_task:
                source_ref = seq_flow.get('sourceRef')
                st = [x for x in self.lrs.get_initial_task(source_ref) if x == key_node][0]
                f = [x.get('id') for x in self.lrs.sequence_flows if x.get('sourceRef') == st and x.get('targetRef') == source_ref][0]
        return f
    
    def _sm3_miner(self):

        print(" -- Mining Process Structure --")
        # Event log file_name
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java']
        if pl.system().lower() != 'windows':
            args.extend(['-Xmx2G', '-Xss8G'])
        args.extend(['-cp',
                        (self.settings['sm3_path']+sep+os.path.join('GenerativeLSTM',
                            'external_tools','splitminer3','lib','*')),
                        'au.edu.unimelb.services.ServiceProvider',
                        'SMD',
                        str(self.settings['epsilon']), str(self.settings['eta']),
                        'false', 'false', 'false',
                        #os.path.join('GenerativeLSTM','input_files', 'spmd', 'ConsultaDataMining201618.xes'),
                        os.path.join('GenerativeLSTM','input_files', 'spmd', self.settings['file'] + '.xes'),
                        self.settings['tobe_bpmn_path'].replace('.bpmn', '')])
        subprocess.call(args)
    
    def _sm3_miner_docker(self, output_path=None):
        print(" -- Mining Process Structure with Dockerized Java 8 + Xvfb --")

        # Classpath for Java inside the container (Linux-style paths)
        classpath = (
            "GenerativeLSTM/external_tools/splitminer3/bpmtk.jar:"
            "GenerativeLSTM/external_tools/splitminer3/lib/*"
        )

        # Prepare input and output paths
        xes_file = os.path.join("GenerativeLSTM", "input_files", "spmd", self.settings["file"] + ".xes").replace("\\", "/")
        if output_path is None:
            output_path = self.settings["tobe_bpmn_path"].replace(".bpmn", "").replace("\\", "/")

        # Host path to mount (make sure it's absolute and Docker-friendly)
        local_path = os.path.abspath(os.getcwd()).replace("\\", "/")

        # Java command to run inside the container
        java_cmd = (
            f'Xvfb :99 -screen 0 1024x768x16 & '
            f'java -cp "{classpath}" '
            f'au.edu.unimelb.services.ServiceProvider '
            f'SMD {self.settings["epsilon"]} {self.settings["eta"]} false false false '
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
