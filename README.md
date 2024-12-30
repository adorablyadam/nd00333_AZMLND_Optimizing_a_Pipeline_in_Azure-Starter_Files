# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

The dataset contained various features related to banking customers. The exercise wasn't explicit nor provided a data dictionary so it is assumed from the file name that based on certain criteria, we seek to predict whether or not a given marketing campaign would be successful based on historical data.

The best performing model for this task involved a trained Voting Ensemble via AutoML, providing an accuracy score of 91.62%.

## Training

Sample Data used for training was available at: https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

Train test split was `80-20`

### Scikit-learn Pipeline

Overall accuracy was `90.88%` with a training time of `16m` for `10 runs`.

#### Hyperparameters
Using HyperDrive, the following hyperparameter space was sampled
- Sampling Method: `RandomParameterSampling`
- Regularization stength: `[ 0.01, 0.1, 1, 10, 100 ]`
- Max Iterations: `[ 100, 200, 300, 400, 500 ]`

#### Early Termination
`Bandit Policy` for early termination was used to allow early stopping of runs if `accuracy` falls significantly below the best performer.

### AutoML

Overall accuracy was `91.62%` with a training time of `28m` for `20 runs`.

Cross Validation: `2`

See run for Ensemble details...

## Pipeline comparison

AutoML slightly outperformed sklearn and does not involve needing to provide hyperparameter sweeping for optimal tuning.

AutoML also provided featurization metrics which could reveal biases in the dataset.

## Future work

Economics of training wasn't fairly evaluated since AutoML ran for twice as many runs. Future work could include running the same number of trials for both cases.
