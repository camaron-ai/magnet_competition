import pandas as pd
import numpy as np
from pathlib import Path
import load_data
from preprocessing.base import preprocessing, create_target
import gc
import logging
import click
import joblib
import mlflow
import default
import os
from metrics import compute_metrics, feature_importances
from models import library as model_library
from pipelines import build_pipeline

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


@click.command()
@click.argument('experiment_path', type=click.Path(exists=True))
@click.option('--eval_mode', type=click.BOOL, default=True)
@click.option('--use_sample', type=click.BOOL, default=False)
@click.option('-m', '--message', type=str, default=None)
def main(experiment_path: str, eval_mode: bool = True,
         use_sample: bool = False,
         message: str = None):
    experiment = os.path.basename(experiment_path)
    logging.info(f'running {experiment}')
    logging.info(f'eval_mode={eval_mode}, use_sample={use_sample}')

    logging.info('reading config file')
    experiment_path = Path(experiment_path)
    config = load_data.read_config_file('./config/config.yml')
    experiment_config = load_data.read_config_file(experiment_path / 'config.yml')

    directories = config['directories']
    data_path = Path(directories['data'])
    prediction_path = experiment_path / 'prediction'
    prediction_path.mkdir(exist_ok=True, parents=True)
    model_path = experiment_path / 'models'
    model_path.mkdir(exist_ok=True, parents=True)

    # reading gt data
    solar_wind_file = ('sample_solar_wind.feather'
                       if use_sample else 'solar_wind.feather')
    logging.info('reading training data')
    dst_labels = load_data.read_csv(data_path / 'dst_labels.csv')
    solar_wind = load_data.read_feather(data_path / solar_wind_file)
    sunspots = load_data.read_csv(data_path / 'sunspots.csv')
    stl_pos = load_data.read_csv(data_path / 'satellite_positions.csv')

    logging.info('applying base preprocessing')
    # applying features pipeline
    data = preprocessing(solar_wind, sunspots, stl_pos,
                         features=default.init_features)
    # create target
    target = create_target(dst_labels)
    assert len(data) == len(target) or use_sample, \
           f'lenght do not match {(len(data), len(target))}'
    # merging
    data = data.merge(target, on=['period', 'timedelta'], how='left')
    data.dropna(subset=['t0', 't1'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    del solar_wind, sunspots, dst_labels
    gc.collect()

    logging.info('splitting dataset')
    train_idx, valid_idx = load_data.split_train_data(data, test_frac=0.2,
                                                      eval_mode=eval_mode)
    train_data = data.loc[train_idx, :]
    valid_data = data.loc[valid_idx, :]

    # importing pipeline
    logging.info('building pipeline')
    pipeline_config = experiment_config.pop('pipeline', {})
    pipeline = build_pipeline(pipeline_config)
    logging.info(f'{pipeline}')

    # fit pipeline
    logging.info('training pipeline')
    pipeline.fit(train_data)
    # transform
    logging.info('transforming datasets')
    train_data = pipeline.transform(train_data)
    valid_data = pipeline.transform(valid_data)

    features = [feature for feature in train_data.columns
                if feature not in default.ignore_features]
    logging.info(f'modeling using {len(features)} features')
    logging.info(f'{features[:30]}')

    # importing model to train
    model_config = experiment_config['model']
    model_instance = model_library[model_config['instance']]
    logging.info('training horizon 0 model')
    # making model for horizon 0
    model_h0 = model_instance(**model_config['parameters'])
    model_h0.fit(train_data.loc[:, features], train_data.loc[:, 't0'])

    logging.info('training horizon 1 model')
    # making model for horizon 1
    model_h1 = model_instance(**model_config['parameters'])
    model_h1.fit(train_data.loc[:, features], train_data.loc[:, 't1'])

    logging.info('prediction h0 and h1 models')
    train_data['yhat_t0'] = model_h0.predict(train_data.loc[:, features])
    train_data['yhat_t1'] = model_h1.predict(train_data.loc[:, features])
    valid_data['yhat_t0'] = model_h0.predict(valid_data.loc[:, features])
    valid_data['yhat_t1'] = model_h1.predict(valid_data.loc[:, features])

    train_error = compute_metrics(train_data, suffix='_train')
    valid_error = compute_metrics(valid_data, suffix='_valid')
    logging.info(f'{train_error}')
    logging.info(f'{valid_error}')
    if eval_mode:
        with mlflow.start_run(run_name=experiment):
            # saving predictions
            train_prediction = train_data.loc[:, default.keep_columns]
            train_prediction.to_csv(prediction_path / 'train.csv', index=False)
            # valid_prediction = valid_data.loc[:, default.keep_columns]
            valid_data.to_csv(prediction_path / 'valid.csv', index=False)
            # saving feature importances if there is aviable
            fi_h0 = feature_importances(model_h0, features)
            fi_h1 = feature_importances(model_h1, features)
            if (fi_h0 is not None) and (fi_h1 is not None):
                fi_h0.to_csv(experiment_path / 'fi_h0.csv', index=False)
                fi_h1.to_csv(experiment_path / 'fi_h1.csv', index=False)
            # saving to mlflow
            # saving metrics
            mlflow.log_metrics(train_error)
            mlflow.log_metrics(valid_error)
            # saving model parameters
            mlflow.log_params(model_config['parameters'])
            tags = {'use_sample': use_sample,
                    'model_instance': model_config['instance'],
                    'experiment': experiment}
            if message is not None:
                tags['message'] = message
            mlflow.set_tags(tags)
    if not eval_mode:
        joblib.dump(model_h0, model_path / 'model_h0.pkl')
        joblib.dump(model_h1, model_path / 'model_h1.pkl')
        joblib.dump(pipeline, model_path / 'pipeline.pkl')


if __name__ == '__main__':
    main()
