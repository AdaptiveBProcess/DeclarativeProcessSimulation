from dataclasses import dataclass
from pathlib import Path
import support_modules.predictor_adapter as pa

@dataclass
class Params:
    root: Path
    log_filename: str
    rep: int = 1
    variant: str = "Rules Based Random Choice"


    @property
    def name(self):
        return self.log_filename.replace(".csv", "")

    @property
    def routes(self):
        base = self.root / "data"
        return {
            "log": base / "0.logs" / self.name,
            "models": base / "1.predicton_models" / self.name,
            "hallucinated": base / "2.hallucination_logs" / self.name,
            "input": base / "2.input_logs" / self.name,
            "bps_asis": base / "3.bps_asis" / self.name,
            "bps_tobe": base / "3.bps_tobe" / self.name,
            "simulation": base / "4.simulation_results" / self.name,
            "rules": base / "0.logs" / self.name / "rules.ini",
            "merged": f"{self.name}_merged.json",
            "bpmn": f"{self.name}.bpmn",
        }

    @property
    def simulation(self):
        return {
            "filename": self.log_filename,
            "log_name": self.name,
            "activity": "pred_log",
            "variant": self.variant,
            "rep": self.rep ,
            "is_single_exec": False,
            "one_timestamp": False,
            "include_org_log": False,
            "model_file": f"{self.name}.h5",
            "folder": pa.get_latest_output_folder(self.routes["models"]),
            "read_options": {
                "timeformat": "%Y-%m-%d %H:%M:%S",
                "one_timestamp": False,
                "filter_d_attrib": False,
                "column_names": {
                    "Case ID": "caseid",
                    "Activity": "task",
                    "lifecycle:transition": "event_type",
                    "Resource": "user",
                },
            },
        }
