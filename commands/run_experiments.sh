# train the nueronet_100
python run_dp_experiment.py ./experiments/bn_neuronet_100;
# remove useless features
python run_dp_experiment.py ./experiments/bn_neuronet_100 -fthres 0.01;
# train final model 
python run_dp_experiment.py ./experiments/bn_neuronet_100 -fthres 0.01 --eval_mode False;


# train the sigmoid_neuronet
python run_dp_experiment.py ./experiments/bn_neuronet_50;
# remove useless features
python run_dp_experiment.py ./experiments/bn_neuronet_50 -fthres 0.01;
# train final model 
python run_dp_experiment.py ./experiments/bn_neuronet_50 -fthres 0.01 --eval_mode False;

 # -------- #

# train the nueronet_100
python run_dp_experiment.py ./experiments/bn_neuronet_100_1L;
# remove useless features
python run_dp_experiment.py ./experiments/bn_neuronet_100_1L -fthres 0.01;
# train final model 
python run_dp_experiment.py ./experiments/bn_neuronet_100_1L -fthres 0.01 --eval_mode False;


# train the sigmoid_neuronet
python run_dp_experiment.py ./experiments/bn_neuronet_25;
# remove useless features
python run_dp_experiment.py ./experiments/bn_neuronet_25 -fthres 0.01;
# train final model 
python run_dp_experiment.py ./experiments/bn_neuronet_25 -fthres 0.01 --eval_mode False;


# train the sigmoid_neuronet
python run_experiment.py ./experiments/lgbm_1200;
# remove useless features
python run_experiment.py ./experiments/lgbm_1200 -fthres 100;
# train final model 
python run_experiment.py ./experiments/lgbm_1200 -fthres 100 --eval_mode False;

python run_experiment.py ./experiments/xgboost;

python run_experiment.py ./experiments/xgboost --eval_mode False;

python ensemble.py;