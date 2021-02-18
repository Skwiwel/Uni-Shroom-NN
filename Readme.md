# Shroom NN

Implementation of a simple neural network from scratch.

Solves the classification problem of shroom edibility based on the [UCI Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/mushroom).

Done as part of a uni project.

## Run instructions

The NN was implemented using Python 3.9.1  
Requires the numpy and scipy libraries:
`python3.9 -m pip install -r requirements.txt`  

The package consists of two subpackages:  
 `train` and `predict`    

Sample command to run the training of the model (assumes the dataset is located in `shroom_nn/resources`):  
```
python3.9 -m shroom_nn.train -t 0.5 -e 1000 --hiddenLayerSize 5  
```

Sample command to run a prediction:  
```
python3.9 -m shroom_nn.predict -i ./shroom_nn/resources/sampleDataForPredictions.data  
```

More detailed run instructions are provided by the subpackages by calling them with the `-h` or `--help` arguments.


## Project structure

All the required scripts and data are located in the `shroom_nn` directory (package root).  
Inside located are two runnable subpackages `train` and `predict` and the packages:
 - `neural` implementing the nn
 - `data` implementing data cleaning functions
 - `utility` providing environment utility functions
 
The directory `resources/` contains sample prediction data, is the default location of the trained model and is meant to be
the directory storing the dataset (not provided so as not to infringe on the dataset donor's rights).

Provided is a simple pre-trained model achieving 100% accuracy on the dataset.
