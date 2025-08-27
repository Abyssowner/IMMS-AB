# IMMS-AB
Use pre trained protein language models to predict antigen antibody affinity.

This library contains the code for the Antigen Antibody Affinity Prediction Model (IMMS-AB), which can be used to predict the antigen antibody affinity of various types of protein complexes.

Before starting, you need to create the environment required for the project. You can choose to create the environment directly using the IMMS-AB.yaml file, or you can create the environment using the requirements.txt file.

## Create Environment with IMMS-AB.yaml
First, create the required environment.
```
cd IMMS-AB
conda env create -f IMMS-AB.yaml
```
Then, activate the "IMMS-AB" environment and enter into the workspace.
```
conda activate IMMS-AB
```
## Usage
The data is in the data folder, the model is in the train folder, and the comparison model directory shows the performance of other models on the same dataset.
