from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import support_modules.predictor_adapter as pa
import pandas as pd

@dataclass
class Params:
    root: Path
    log_filename: str
    rep: int = 1
    variant: str = "Rules Based Random Choice"
    rules_filename: str = "rules_global.ini"
    tobe_cases: Optional[int] = None  # None = usa total_cases (poblacion real del log)
    hallucination_cases: Optional[int] = None  # None = usa total_cases del log original

    @property
    def total_cases(self) -> int:
        """Número de casos únicos en el log original — usado como total_cases en Prosimos."""
        log_path = self.routes["log"] / self.log_filename
        df = pd.read_csv(log_path, usecols=["case:concept:name"])
        return df["case:concept:name"].nunique()

    @property
    def effective_tobe_cases(self) -> int:
        """Casos a simular en Prosimos -- tobe_cases explicito, o total_cases si no se fija.

        Antes `tobe_cases` tenia un default fijo de 371 (el tamano de muestra de Cochran
        calculado para BPIC12), aplicado sin querer a cualquier log. Con `None` como default,
        cada log usa su propia poblacion real salvo que se pida explicitamente otro numero
        (ej. un override chico para pruebas rapidas de humo, o el 371 de Cochran para BPIC12).
        """
        return self.tobe_cases if self.tobe_cases is not None else self.total_cases


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
            "rules": base / "0.logs" / self.name / self.rules_filename,
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
            "rep": self.rep,
            "num_instances": self.hallucination_cases if self.hallucination_cases is not None else self.total_cases,
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
