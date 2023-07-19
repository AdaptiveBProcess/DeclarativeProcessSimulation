# Automated Generation of Process Simulation Scenarios from Declarative Control-Flow Changes

The code here presented is able to execute different pre- and post-processing methods and architectures for building and using generative models from event logs in XES format using LSTM anf GRU neural networks. This code can perform the next tasks:

* Training LSTM neuronal networks using an event log as input.
* Generate full event logs using a trained LSTM neuronal network.
* Generate traces that adhere to a corresponding set of rules given by the user.
* Predict the remaining time and the continuation (suffix) of an incomplete business process trace.
* Discover the stochastic process model and generate a simulation based on the rules given by the user.
## Architecture

![alt text](https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation/blob/main/images/Pipeline%202.png)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

```
git clone https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation.git
```

### Prerequisites

To execute this code you just need to install Anaconda in your system and create an environment using the *environment.yml* specification provided in the repository.
```
cd GenerativeLSTM
conda env create -f environment.yml
conda activate deep_generator
```

## Running the script

### Training the model 
Train the model with the input event log.
```
python dg_training.py -f event-log-name.xes
```
This generates a folder in output_files. Copy that folder name into dg_prediction.py and replace the value of the variable parameters['folder'].

### Generate predictions
Train the model with the input event log.
```
python dg_prediction.py
```
This generates a simulation process model corresponding to the implementation of the changes proposed by the user. The simulation files are stored in output_files/simulation_files/. In addition, the approach simulates that simulation process model and generates a statistics file corresponding to the stats of the simulated model. The simulation stats are stored in output_files/simulation_stats/.

## Examples

## Authors

* **Daniel Baron**
* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
