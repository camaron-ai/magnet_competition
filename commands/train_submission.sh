# train the nueronet_100
python run_dp_experiment.py ./experiments/neuronet_100;
# remove useless features
python run_dp_experiment.py ./experiments/neuronet_100 -fthres 0.;
# train final model 
# python run_dp_experiment.py ./experiments/neuronet_100 -fthres 0. --eval_mode False;


# train the sigmoid_neuronet
python run_dp_experiment.py ./experiments/sigmoid_neuronet;
# remove useless features
python run_dp_experiment.py ./experiments/sigmoid_neuronet -fthres 0.;
# train final model 
# python run_dp_experiment.py ./experiments/sigmoid_neuronet -fthres 0. --eval_mode False;


# train the sigmoid_neuronet
python run_experiment.py ./experiments/lgbm_1200;
# remove useless features
python run_experiment.py ./experiments/lgbm_1200 -fthres 100.;
# train final model 
# python run_experiment.py ./experiments/lgbm_1200 -fthres 100. --eval_mode False;