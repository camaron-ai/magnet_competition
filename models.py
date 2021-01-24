from sklearn.linear_model import LinearRegression, Lasso
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor

library = {'LR': LinearRegression,
           'LGBM': LGBMRegressor,
           'RF': RandomForestRegressor,
           'LASSO': Lasso}
        #    'XGBOOST': XGBRegressor}

