# Automated Generation of Process Simulation Scenarios from Declarative Control-Flow Changes

This project enables automated training and generation of business process simulation scenarios based on declarative control-flow changes using deep learning (LSTM/GRU) models trained on event logs (in CSV format). It includes modules for training, prediction, rule-based simulation, and end-to-end pipeline execution.

---

## 💡 Main Features

* Train LSTM/GRU models using event logs.
* Generate full event logs using trained models.
* Apply declarative rules to constrain trace generation.
* Predict suffix and remaining time of incomplete traces.
* Discover BPMN process models and simulate their execution.

---

## 🧱 Architecture Overview

![Pipeline](https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation/blob/main/docs/pipeline/Pipeline.png)

---

## 🗂️ New Modular Folder Structure

```
data/
├───0.logs/                        # Raw event logs and rules
│   └───<log_name>/embedded_matix 
├───1.predicton_models/           # Trained models
│   └───<log_name>/<model_folder>/parameters/traces_generated
├───2.hallucination_logs/         # Generated synthetic traces
├───2.input_logs/                 # Preprocessed input logs
├───3.bps_asis/                   # BPMN models discovered (as-is)
│   └───<log_name>/<run_id>/best_result
├───3.bps_tobe/                   # BPMN models simulated (to-be)
│   └───<log_name>/<run_id>/best_result
└───4.simulation_results/         # Simulation statistics
    └───<log_name>/<rule_applied>
```

---

## ⚙️ System Requirements

* Python 3.x
* Java SDK 1.8 (compatible version for your OS)
* Anaconda Distribution
* Git

---

## 🚀 Getting Started

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation.git
cd DeclarativeProcessSimulation
git submodule update --init --recursive
```

Checkout specific branches:

```bash
cd GenerativeLSTM 
git checkout Declarative-Process
cd ..

cd Simod-2.3.1
git checkout v2.3.1
cd ..
```

### Set up the environment

```bash
cd GenerativeLSTM
conda env create -f environment.yml
conda activate deep_generator
```

Create the following folders if not already present:

```bash
mkdir -p data/0.logs
mkdir -p data/1.predicton_models
mkdir -p data/2.hallucination_logs
mkdir -p data/2.input_logs
mkdir -p data/3.bps_asis
mkdir -p data/3.bps_tobe
mkdir -p data/4.simulation_results

```

---

## 🧪 Running the Pipeline

### 1. Place your event log

Put the `.csv` log into:

```bash
0.logs/<log_name>/
```

Modify `dg_training.py` to set the log name and model settings (line \~43):

```python
parameters['filename'] = 'your_log_name.csv'
```

Run training:

```bash
python dg_training.py
```

This will generate a new folder inside `1.predicton_models/<log_name>/`.

### 2. Prediction

In `dg_prediction.py`, update the folder and filename (lines \~140 and \~159):

```python
parameters['filename'] = 'your_log_name.csv'
parameters['folder'] = 'YYYYMMDD_XXXXXXXX_XXXX_XXXX_XXXX_XXXXXXXXXXX'
```

Specify declarative rules in `rules.ini`, such as:

```ini
# Example Rule
[RULES]
path = TaskA >> TaskB
variation = =1
```

Run:

```bash
python dg_prediction.py
```

Output:

* Synthetic traces → `2.hallucination_logs/<log_name>/`
* BPMN model (To-Be) → `3.bps_tobe/<log_name>/...`
* Simulation results → `4.simulation_results/<log_name>/...`

---

## 📁 File Definitions

* `0.logs`: Input CSV logs.
* `1.predicton_models`: Model checkpoints and generated traces.
* `2.input_logs`: Intermediate preprocessed logs.
* `2.hallucination_logs`: Traces generated using trained models and rules.
* `3.bps_asis`: Discovered BPMN models from original logs.
* `3.bps_tobe`: BPMN models generated from hallucinated logs.
* `4.simulation_results`: Stats from simulations based on the BPMN models.

---

## 🧪 Examples

Check `input_files` for example logs and `rules.ini` for declarative rule formats.

---

## 👤 Authors

* **David Sequera**
* **Daniel Baron**
* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
