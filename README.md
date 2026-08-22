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

This forlder is in the root of the project

```
data/
├───0.logs/                        # Raw event logs and rules
│   └───<log_name>/embedded_matix
│   └───<log_name>/<log_name>.csv
│   └───<log_name>/rule.ini
├───1.predicton_models/           # Trained models
│   └───<log_name>/<model_folder>/parameters/traces_generated
├───2.hallucination_logs/         # Generated synthetic traces
├───2.input_logs/                 # Preprocessed input logs
├───3.bps_asis/                   # BPMN models discovered (as-is)
│   └───<log_name>/<run_id>/best_result
├───3.bps_tobe/                   # BPMN models simulated (to-be)
│   └───<log_name>/<run_id>/best_result
├───4.simulation_results/         # Simulation statistics
│   └───<log_name>/<rule_applied>
└───5.rule_selection/             # GENESIS pipeline outputs (see below)
    └───<log_name>/{uplift_ranking,screening_summary,final_comparison}.csv
```

---

## ⚙️ System Requirements

* Python 3.x
* Java SDK 1.8 (compatible version for your OS)
* Anaconda Distribution
* Git

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation.git
cd DeclarativeProcessSimulation
```

(`GenerativeLSTM/` used to be a separate submodule repository; it is now tracked
directly inside this repository, so no submodule setup is needed.)

### Set up the environment

```bash
cd GenerativeLSTM
conda env create -f environment.yml
conda activate deep_generator
```

Create the following folders if not already present in the root of the project:

```bash
mkdir -p data/0.logs
mkdir -p data/1.predicton_models
mkdir -p data/2.hallucination_logs
mkdir -p data/2.input_logs
mkdir -p data/3.bps_asis
mkdir -p data/3.bps_tobe
mkdir -p data/4.simulation_results
mkdir -p data/5.rule_selection

```

### Set up docker images


```bash
docker pull nokal/simod
docker build -t java8-xvfb docs/example/java_docker_image
docker image ls
```

---

## 🧪 Preparing a New Log

One-time setup before training a model or running the GENESIS pipeline on a new event
log for the first time:

### 1. Place the event log

```bash
data/0.logs/<log_name>/<log_name>.csv
```

### 2. Add the Simod configuration files

Required by the simulation stages (Simod's discovery of the AS-IS and TO-BE process
models) — these must exist beforehand, they are not generated automatically. One
command writes both from the same template (`docs/example/configuration.yaml`),
keyed only by the log name, so the two copies never drift apart from a manual
copy-paste mistake:

```bash
python -m rule_selection.setup_log_config --log <log_name>.csv
```

This writes `data/2.input_logs/<log_name>/configuration_original.yaml` and
`data/2.hallucination_logs/<log_name>/configuration_generated.yaml` — identical except
for the `train_log_path` line, which each one points at its own log. It skips a file
that already exists; pass `--force` to overwrite. If the log uses different column
names or conventions than `caseid/task/user/start_timestamp/end_timestamp`, edit the
generated files' `log_ids` section by hand afterward — this command only automates the
copy with the correct log name, not adapting the content to a differently-structured log.

That's it — no declarative rule needs to be written by hand. The GENESIS pipeline below
discovers, ranks, formalizes, and simulates candidate rules automatically; nothing about
the control-flow policy is defined manually.

---

## 🎯 GENESIS Pipeline (`rule_selection/`)

Discovers, recommends, and evaluates by simulation a declarative control-flow policy
to improve a business process's time performance. Given an event log, the pipeline
goes through 5 stages: it discovers candidate constraints, ranks them by statistical
impact, formalizes the best one as a simulatable rule, picks the winner by simulating
each candidate, and finally compares it against 3 baselines (AS-IS, Placebo,
Top-Support).

### Requirements

- Active `deep_generator` conda environment (`environment.yml` at the repo root).
- Java (for MINERful — constraint discovery).
- Docker Desktop running (for Simod and Prosimos — process model discovery and
  simulation).
- The log and its Simod configuration YAMLs already prepared (see "Preparing a New
  Log" above).

### The 5 stages

| # | Stage | What it does | What it produces |
|---|---|---|---|
| 1 | **Constraint discovery** | Runs MINERful once on the full log | Every candidate Declare constraint (`candidates_all.csv`, `candidates_supported.csv`) |
| 2 | **Statistical impact estimation** | For each supported candidate, computes the PCE uplift (compliant vs. non-compliant) and a Welch score (lower bound of the 95% confidence interval) | Ranking of the best candidates (`uplift_ranking.csv`) |
| 3 | **Formal rule specification** | Translates each ranked candidate into the `.ini` syntax the simulator consumes | One `.ini` file per candidate, in `data/0.logs/<log>/` |
| 4 | **Simulation-based selection** | Fully simulates each ranked candidate (LSTM hallucination → Simod → Prosimos) and picks the one with the highest real simulated PCE | `screening_summary.csv` (winner = row with the highest PCE) |
| 5 | **Final comparison (arms)** | Simulates 4 scenarios with the same number of replicas — AS-IS (real log), Placebo (generator with no rule at all), Top-Support (highest-frequency rule), and GENESIS (the winner from stage 4) — and computes each one's causal effect | `final_comparison.csv`, with the Δ_generator and Δ_policy deltas |

Stage 1 requires an LSTM model already trained for the log:

```bash
python dg_training.py -f RunningExample.csv
```

(Not one of the method's 5 stages itself, just a prerequisite. If you use Option A
below, this step is optional: `run_full_pipeline.py` trains the model automatically as
its first step if none exists yet for the log — this can take hours with the default
settings.)

### How to run it: two options

#### Option A — Everything at once (`run_full_pipeline.py`)

A single command runs all 5 stages end to end. The main input is `--log`, the CSV
filename of the event log to run the whole pipeline against (e.g.
`--log RunningExample.csv`).

Before running it, make sure you have:

- **The event log**, at `data/0.logs/<log_name>/<log_name>.csv`.
- **The Simod configuration YAMLs**, at `data/2.input_logs/<log_name>/` and
  `data/2.hallucination_logs/<log_name>/` (see "Preparing a New Log" above — these are
  not generated on their own).
- **An LSTM model trained for the log** — optional: if none exists,
  `run_full_pipeline.py` trains it automatically as its first step (can take hours
  with the default settings).

This is the recommended way to run a full pass: every stage (and every
candidate/arm within stages 4 and 5) is launched in its own process, so if something
gets interrupted partway through (Docker fails, the machine crashes),
**re-running the same command resumes where it left off** — no need to retrain the
model or resimulate what already finished.

```bash
python -m rule_selection.run_full_pipeline --log RunningExample.csv
```

With reduced population and epochs, for a quick test run:

```bash
python -m rule_selection.run_full_pipeline --log RunningExample.csv \
    --hallucination-cases 30 --tobe-cases 30 --epochs 2 --max-eval 1 \
    --screening-replicas 2 --final-replicas 2
```

If you need to redo something specific (e.g., a screening candidate suspected of
having corrupted data) without repeating everything else:

```bash
python -m rule_selection.run_full_pipeline --log RunningExample.csv \
    --force-rescreen notchainsuccession_Task_C_Task_B
```

#### Option B — Step by step (individual modules)

To closely supervise each stage, test a single candidate or a single arm before
committing to a full run, or diagnose a specific issue without re-running the whole
pipeline:

```bash
# 1-3. Train the model (if one doesn't already exist for this log)
python dg_training.py -f RunningExample.csv

# 1-2. Discovery + ranking (no Docker needed, only Java) -- computes the ranking
# (uplift_ranking.csv) but does NOT write the candidates' .ini files yet
python -m rule_selection.run_ranking --log RunningExample.csv

# Test ONE single candidate end to end (useful to verify everything works
# before running all 5) -- the .ini filename is always "<candidate_id>.ini"
# (candidate_id comes from uplift_ranking.csv); if that file doesn't exist yet,
# run_screening.py/run_full_pipeline.py write it automatically while processing
# that candidate -- it never needs to be created by hand
python -m rule_selection.manual_arm_test --log RunningExample.csv \
    --rule notchainsuccession_Task_C_Task_B.ini --replicas 2

# 4. Screening of the ranking's 5 candidates
python -m rule_selection.run_screening --log RunningExample.csv

# 5. Final comparison, one arm at a time (asis, placebo, topsupport, genesis)
python -m rule_selection.run_final_arm --log RunningExample.csv --arm asis \
    --asis-run-label <label_of_the_already_discovered_asis> --replicas 20
python -m rule_selection.run_final_arm --log RunningExample.csv --arm placebo \
    --asis-run-label <label_of_the_already_discovered_asis> --replicas 20
python -m rule_selection.run_final_arm --log RunningExample.csv --arm topsupport \
    --asis-run-label <label_of_the_already_discovered_asis> --replicas 20
python -m rule_selection.run_final_arm --log RunningExample.csv --arm genesis --replicas 20

# Consolidate the 4 arms into final_comparison.csv
python -m rule_selection.run_final_comparison_consolidate --log RunningExample.csv
```

All of the commands above accept `--hallucination-cases`/`--tobe-cases` for a small
test run (by default they use the log's real population).

#### Which one to pick?

- **Real run for the paper, small/medium log** (RunningExample, PurchasingExample):
  Option A, without any reduced-population flag.
- **Large log** (BPI_Challenge_2012): Option A works too (every candidate/arm already
  runs in its own process), but it's worth watching the progress stage by stage the
  first time.
- **You're verifying the pipeline works, or debugging a specific issue**: Option B —
  run just the piece you care about.

### Where results end up

All under `data/5.rule_selection/<log_name>/`:

| File | Contents |
|---|---|
| `candidates_all.csv` / `candidates_supported.csv` | Every discovered constraint / the ones the simulator can impose |
| `uplift_ranking.csv` | Ranking by Welch score |
| `screening_summary.csv` | Simulated PCE of each ranked candidate (winner = row 0) |
| `final_comparison.csv` | The 4 arms with their causal deltas — the final result that goes into the paper |

---

## 📁 File Definitions

* `0.logs`: Input CSV logs.
* `1.predicton_models`: Model checkpoints and generated traces.
* `2.input_logs`: Intermediate preprocessed logs (here is important to have a `configuration.yaml` ).
* `2.hallucination_logs`: Traces generated using trained models and rules (here is important to have a `configuration.yaml` ).
* `3.bps_asis`: Discovered BPMN models from original logs.
* `3.bps_tobe`: BPMN models generated from hallucinated logs.
* `4.simulation_results`: Stats from simulations based on the BPMN models.
* `5.rule_selection`: Outputs of the GENESIS pipeline (`rule_selection/`) — candidate
  ranking, screening summary, and final 4-arm comparison. See the section above.

---

## 🧪 Examples

Check `docs/example` for reference logs and `configuration.yaml` files. `rules.ini` there
shows the format the GENESIS pipeline itself writes for each candidate rule — useful for
understanding the format or for debugging, not something you need to author by hand.

---


## ❌ Known Issues

* `BIMP`: BIMP log version is not beeing created. This happends because Promious creates a log for each resource but bimp needs a role (we are trying to cast this business logic)

---


## 👤 Authors

* **David Sequera**
* **Daniel Baron**
* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
