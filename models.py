from sklearn.linear_model import LinearRegression, Lasso
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from dplr.model import SimpleDeepNet, Resnet

library = {'LR': LinearRegression,
           'LGBM': LGBMRegressor,
           'RF': RandomForestRegressor,
           'LASSO': Lasso,
           'simple_deep_net': SimpleDeepNet,
           'resnet': Resnet}
        #    'XGBOOST': XGBRegressor}

