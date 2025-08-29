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

The specific training process is described in train.py, while model. py represents the model structure and provides a Jupyter notebook version for your use. Jupyter notebook provides complete data segmentation, model training, and testing processes.

The following are the functions of the input parameters of the model：

```
num_features = number of features, also known as the d_model parameter of Transformer
num_classes = number of categories.
nhead = number of heads in Transformer
num_coder_layers = The number of layers in the Transformer encoder
num_decoder_layers = the number of layers in the Transformer decoder
learning_rate = learning rate
seed = random seed
train_tsv = dataset used for training and testing
```

The predict.py provides a method for predicting the affinity of a single antigen antibody sequence pair, with the output being the affinity between the two.
