# Automated Generation of Process Simulation Scenarios from Declarative Control-Flow Changes

The code here presented is able to execute different pre- and post-processing methods and architectures for building and using generative models from event logs in XES format using LSTM anf GRU neural networks. This code can perform the next tasks:

* Training LSTM neuronal networks using an event log as input.
* Generate full event logs using a trained LSTM neuronal network.
* Generate traces that adhere to a corresponding set of rules given by the user.
* Predict the remaining time and the continuation (suffix) of an incomplete business process trace.
* Discover the stochastic process model and generate a simulation based on the rules given by the user.
## Architecture


![alt text](https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation/blob/main/images/Pipeline%202.png)

## System Requirements
* Python 3.x
* Java SDK 1.8 Choose right version according with Operative System.
* Anaconda Distribution
* Git

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

```
git clone --recurse-submodules https://github.com/AdaptiveBProcess/DeclarativeProcessSimulation.git
```
Once the repository has been cloned, you can update the submodules at any time using the following command:

```
git submodule update --init --recursive

```
For Simod it is necessary to update version with this commands

```
cd Simod-2.3.1
git checkout Declarative-Process
cd ..
Generative LSTM branch

```
cd GenerativeLSTM
git checkout v2.3.1
cd ..

```
### Prerequisites

To execute this code with the previous Anaconda install in your system, create an environment using the *environment.yml* specification provided in the repository.
```
cd GenerativeLSTM
conda env create -f environment.yml
conda activate deep_generator
```
Be sure when running this script to be using Conda prompt or to configure conda into another prompt

Here is an example here for adding conda to Windows prompt in vs-code if needed
https://stackoverflow.com/questions/54828713/working-with-anaconda-in-visual-studio-code

## Running the script

### Training the model 
Train the model with the input event log.
```
python dg_training.py -f {event-log-name}.xes
```

This generates a folder in output_files. **Copy that folder name into dg_prediction.py and replace the value of the variable parameters['folder'].**

### Generate predictions
For this step Java 1.8 SDK is needed.

Train the model with the input event log.

Before starting this step in the route \GenerativeLSTM\output_files please create a folder named \GenerativeLSTM\output_files\simulation_files for the program to find the exact route.

When training the model, be sure to use the appropiate rules related to the BPMN model that is being used. **Rules.ini** gives an idea of which rules can be used to ensure simulation data goes as well as posible
```
python dg_prediction.py
```
This generates a simulation process model corresponding to the implementation of the changes proposed by the user. The simulation files are stored in output_files/simulation_files/. In addition, the approach simulates that simulation process model and generates a statistics file corresponding to the stats of the simulated model. The simulation stats are stored in output_files/simulation_stats/.


## Examples
The files used for the experimentation are stored in input_files.


## Authors

* **Daniel Baron**
* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**