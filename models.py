from sklearn.linear_model import LinearRegression, Lasso
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

library = {'LR': LinearRegression,
           'LGBM': LGBMRegressor,
           'RF': RandomForestRegressor,
           'LASSO': Lasso}
