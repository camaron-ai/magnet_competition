experiment_folder='./experiments/*'
for experiment in $experiment_folder
do
    echo "running $experiment";
    python run_experiment.py $experiment $@;
done