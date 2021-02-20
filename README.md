# NOAA Challenge 3rd Place Solution

Welcome to the 3rd place solution for the [MagNet: Model the Geomagnetic Field](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/) competition.
This repository contains everything neccesary to replicate our solution.

## (0) Getting started

### Prerequisites
You can install all neccesary dependecies on your own machines with conda (highly recommended) or with pip.
if you are using Anaconda then run:
```bash
conda env create -f environment.yml
conda activate magnet
```
if you are using pip then run:
```bash
conda env create -f environment.yml
source activate magnet
```

### Download the Data

First, **download the data** from the competition [download page](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/data/)
and put each file in the `data/raw/` folder. After you get the data, you should have
these files:

```
data/raw/
├── dst_labels.csv
├── satellite_positions.csv
├── solar_wind.csv
└── sunspots.csv
```

## (1) Compute Features
To compute the training dataset, run the following command
```bash
bash commands/compute_dataset.sh --n_jobs {n_jobs}
```
where n_jobs is the number of jobs to run in parallel. the default value is 1

in resumen, this command will:
- save the solar wind CSV file as a Feather file.
- Apply the feature engineering pipeline to the time series

after running the commands, you should have these files:
```
data/interim/
├──  solar_wind.feather
data/processed/
├──  fe.feather
```
this step may take sometime because the solar wind data is very large and we tried to simulate the evaluation process where we predict each timedelta at a time. It is important to use multiple cores to speed things up.

## (2) Train and Validate Experiments

### Solution Overview

Our solution is an ensemble of 3 models, 1 LGBM and 2 feed-forward neuronets with dropout and batch normalization, you can find the specific parameters of such models in the models_config/ folder. In the case of the LGBM we train 2 models, one for each horizon (t and t + 1 hour) but for the feed-forward neuronet we train only one model.

We compute a lot of features and most of then are useless or redundant, that's why for each model we do:
- Train the model with all features
- Calculate the feature importance
- Train the model again but this time only with important features

This approach helped us to reduce overfit, improve our validation score and reduce the complexity of our model.

### Running Models
to run all the steps above, run the following command:
```bash
bash commands/train_and_validate.sh
```
for each of the model, this command will:
- Validate the Model using 20% of the data
- Train the final model using all data available

After completing this step, you are ready to do inference!

### Metrics And Artifacts
The scores and parameters of each models are registered using Mlflow, if you want to look at the results open the mlflow web UI with the command below and then open your browser at `http://localhost:<mlflow_port>`
```bash
mlflow ui
```
We save more information about the training and scores, this additional information is saved in the experiment's folder.

### Ensemble
Once you have trained all experiments, the next step is to look how much the score improve by ensembling, we approach this by finding the subset of experiments that has the best ensemble score. We do this by executing the ensemble.py script:
```bash
python src/ensemble.py --upto {upto}
```
where the upto parameter is the maximum number of experiments that can be ensemble. the defualt value is equal to the number of experiments in the experiments/ folder, in other words, it will compute the ensemble score for every possible subset. Our solution is compose only for 3 models so there is not need for specifying the upto parameter, nevertheless, it is important when the number of experiment is bigger because it may take a lot of time to run.

The script will create a CSV file as below:
|   h0_rmse |   h1_rmse |    rmse | experiment                                |   n_model |
 |----------:|----------:|--------:|:------------------------------------------|----------:|
 |   9.54384 |   9.57683 | 9.56035 | lgbm_1200__neuronet_100__sigmoid_neuronet |         3 |
 |   9.61142 |   9.69147 | 9.65153 | lgbm_1200__neuronet_100                   |         2 |
 |   9.68632 |   9.68409 | 9.6852  | lgbm_1200__sigmoid_neuronet               |         2 |
 |   9.739   |   9.77314 | 9.75608 | neuronet_100__sigmoid_neuronet            |         2 |

this CSV file will be save in the following path:
```
experiments/
├──  ensemble_summary.csv
```



