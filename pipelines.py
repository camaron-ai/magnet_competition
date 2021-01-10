from typing import Dict, Any
from preprocessing.transformer import DropFeatureByCorr
from preprocessing.transformer import DropFeaturesByCorrTarget
from preprocessing.transformer import Lagger
from preprocessing.transformer import RollingStats
from preprocessing.transformer import NoOp
from preprocessing.transformer import DifferenceFeatures
from sklearn.pipeline import Pipeline


library = {'drop_feature_corr': DropFeatureByCorr,
           'drop_feature_target': DropFeaturesByCorrTarget,
           'lagger': Lagger,
           'rolling_stats': RollingStats,
           'difference': DifferenceFeatures}


def build_pipeline(pipeline_config: Dict[str, Dict[str, Any]]):
    if len(pipeline_config) == 0:
        return Pipeline(steps=[('no_op', NoOp())])
    steps = []
    for pipeline_name, pipeline_config in pipeline_config.items():
        pipeline_instance = library[pipeline_name]
        steps.append((pipeline_name, pipeline_instance(**pipeline_config)))
    return Pipeline(steps=steps)


def set_common_params(pipeline, **kargs):
    params = {f'{step}__{name}': value
              for step, transformer in pipeline.named_steps.items()
              for name, value in kargs.items()
              if name in transformer.get_params()}
    pipeline.set_params(**params)
