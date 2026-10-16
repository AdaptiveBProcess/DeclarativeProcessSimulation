"""Checks that our DECLARE rules reproduce the condition labels used by CVAE
(Graziosi et al., 2024): for every case of a CVAE dataset, the rule must hold
exactly when the case is labelled 'deviant'.

Datasets: https://github.com/rgraziosi-fbk/cvae-process-mining/tree/main/datasets
Usage (from the repository root):
    python validation/validar_plantillas_vs_cvae.py <cvae_csv> <rules_ini>
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support_modules import traces_evaluation as te  # noqa: E402


def load_cvae_log(csv_path):
    log = pd.read_csv(csv_path, sep=';', low_memory=False)
    case_col = 'case:concept:name' if 'case:concept:name' in log.columns else 'Case ID'
    act_col = 'concept:name' if 'concept:name' in log.columns else 'Activity'
    return pd.DataFrame({
        'caseid': log[case_col].astype(str),
        'task': log[act_col].astype(str),
        'start_timestamp': pd.to_datetime(log['time:timestamp']),
        'label': log['label'],
    })


def main(csv_path, rules_path):
    rules = te.extract_rules(path=rules_path)
    log = load_cvae_log(csv_path)
    rows = []
    for caseid, case in log.groupby('caseid', sort=False):
        case = case.sort_values('start_timestamp', kind='mergesort')
        predicted = te.case_satisfies_rule(case, rules['path'], rules['rule'])
        rows.append((case['label'].iloc[0] == 'deviant', predicted))
    result = pd.DataFrame(rows, columns=['cvae_deviant', 'rule_holds'])
    agreement = (result['cvae_deviant'] == result['rule_holds']).mean()
    print(f"{os.path.basename(csv_path)} | rule={rules['rule']} {rules['path']}")
    print(f"  cases={len(result)}  deviant={result['cvae_deviant'].mean():.2%}  "
          f"rule holds={result['rule_holds'].mean():.2%}  agreement={agreement:.2%}")
    print(pd.crosstab(result['cvae_deviant'], result['rule_holds']))
    return agreement


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) == 1.0 else 1)
