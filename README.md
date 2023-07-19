# Automated Generation of Process Simulation Scenarios from Declarative Control-Flow Changes

The code here presented is able to execute different pre- and post-processing methods and architectures for building and using generative models from event logs in XES format using LSTM anf GRU neural networks. This code can perform the next tasks:

* Training LSTM neuronal networks using an event log as input.
* Generate full event logs using a trained LSTM neuronal network.
* Generate traces that adhere to a corresponding set of rules given by the user.
* Predict the remaining time and the continuation (suffix) of an incomplete business process trace.
* Discover the stochastic process model and generate a simulation based on the rules given by the user.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

```
git clone [https://github.com/AdaptiveBProcess/GenerativeLSTM.git](https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation.git)
```

### Prerequisites

To execute this code you just need to install Anaconda in your system and create an environment using the *environment.yml* specification provided in the repository.
```
cd GenerativeLSTM
conda env create -f environment.yml
conda activate deep_generator
```

## Running the script

## Examples

## Authors

* **Daniel Baron**
* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
