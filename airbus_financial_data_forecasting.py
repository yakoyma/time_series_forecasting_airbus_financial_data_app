"""
===============================================================================
Time Series Project: Airbus Financial Data Forecasting Application
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
"""
# Standard libraries
import random
import platform

# Other libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sktime
import sklearn
import joblib
import catboost
import lightgbm
import xgboost


from sktime.utils import mlflow_sktime
from sktime.utils.plotting import plot_series
from sktime.split import temporal_train_test_split, ExpandingWindowSplitter
from sktime.forecasting.compose import MultiplexForecaster, make_reduction
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.var import VAR
from sktime.forecasting.neuralprophet import NeuralProphet
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              ExtraTreesRegressor)
from joblib import dump
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from functions import *


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Joblib: {}'.format(joblib.__version__))
print('Catboost: {}'.format(catboost.__version__))
print('Lightgbm: {}'.format(lightgbm.__version__))
print('Xgboost: {}'.format(xgboost.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
SP = 5
FORECASTING_HORIZON = 20

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print('\n\n\n1. Load the dataset')


# Load the dataset
INPUT_CSV = 'datasets/airbus_financial_data.csv'
raw_dataset = load_dataset(file_path=INPUT_CSV, encoding='utf-8')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print('\n\n\n2. Feature Engineering')


# Feature selection
X = raw_dataset.filter(['Date', 'Close', 'High', 'Low', 'Volume'])


# Create the target (the daily logarithmic returns)
X['Target'] = np.log(X['Close'] / X['Close'].shift(1))


# Management of missing data
if X.isna().any().any() == True:
    X = X.dropna()
    X.reset_index(inplace=True, drop=True)


# Retrieve the target
X = X.set_index('Date')
X.index = pd.PeriodIndex(data=X.index, freq='B')
y = X.filter(['Target'])


# Create the high minus low indicator (HML) feature
X['HML'] = X['High'] - X['Low']


# Create the relative volume indicator (RVOL) feature
X['RVOL'] = X['Volume'] / X['Volume'].rolling(
    window=50, min_periods=1).mean()
X['RVOL'] = X['RVOL'].fillna(0)


# Split the dataset into train and test sets
y_train, y_test, X_train, X_test = temporal_train_test_split(
    y=y, X=X, test_size=FORECASTING_HORIZON)


# Correlation analysis
corr_coef = X_train['HML'].corr(X_train['RVOL'])
print(f'\n\nCorrelation coefficient between HML and RVOL '
      f'features: {corr_coef:.3f}')


# Save the test set (reference dataset) in CSV format
reference_dataset = X_test.assign(Target=y_test)
OUTPUT_CSV_1 = 'datasets/reference_dataset.csv'
reference_dataset.reset_index().to_csv(OUTPUT_CSV_1, index=False)


# Normalisation
scaler = StandardScaler()
X_train = pd.DataFrame(
    data=scaler.fit_transform(X_train[['HML', 'RVOL']]),
    index=X_train.index,
    columns=['HML', 'RVOL']
)
X_test = pd.DataFrame(
    data=scaler.transform(X_test[['HML', 'RVOL']]),
    index=X_test.index,
    columns=['HML', 'RVOL']
)

# Scaler persistence
dump(value=scaler, filename='models/scaler/scaler.joblib')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print('\n\n\n3. Machine Learning')


# Instantiate the model
estimators = MultiplexForecaster(
    forecasters=[
        ('DummyGlobalForecaster', DummyGlobalForecaster()),
        ('NaiveForecaster', NaiveForecaster(sp=SP)),
        ('VAR', VAR(maxlags=10, freq='B', random_state=SEED)),
        ('NeuralProphet', NeuralProphet(growth='off', freq='B')),
        ('TinyTimeMixerForecaster', TinyTimeMixerForecaster(freq='B')),
        ('LinearRegression',
             make_reduction(estimator=LinearRegression(n_jobs=-1))
         ),
        ('KNeighborsRegressor',
             make_reduction(
                 estimator=KNeighborsRegressor(n_neighbors=1, n_jobs=-1)
             )
         ),
        ('RandomForestRegressor',
             make_reduction(
                 estimator=RandomForestRegressor(n_jobs=-1, random_state=SEED)
             )
         ),
        ('GradientBoostingRegressor',
             make_reduction(
                 estimator=GradientBoostingRegressor(random_state=SEED)
             )
         ),
        ('ExtraTreesRegressor',
             make_reduction(
                 estimator=ExtraTreesRegressor(n_jobs=-1, random_state=SEED)
             )
         ),
        ('CatBoostRegressor',
             make_reduction(estimator=CatBoostRegressor(random_seed=SEED))
         ),
        ('LGBMRegressor',
             make_reduction(
                 estimator=LGBMRegressor(random_state=SEED, n_jobs=-1)
             )
         ),
        ('XGBRegressor',
             make_reduction(estimator=XGBRegressor())
         ),
    ]
)
cv = ExpandingWindowSplitter(initial_window=5 * 252, step_length=252)
model = ForecastingGridSearchCV(
    forecaster=estimators,
    cv=cv,
    param_grid={
        'selected_forecaster':
            [
                'DummyGlobalForecaster',
                'NaiveForecaster',
                'VAR',
                'NeuralProphet',
                'TinyTimeMixerForecaster',
                'LinearRegression',
                'KNeighborsRegressor',
                'RandomForestRegressor',
                'GradientBoostingRegressor',
                'ExtraTreesRegressor',
                'CatBoostRegressor',
                'LGBMRegressor',
                'XGBRegressor'
            ]
    },
    scoring=MeanAbsoluteError()
)

# Train the model
model.fit(y=y_train, X=X_train)

# Display the best model
print(f'\nBest forecaster: {model.best_forecaster_}')
print(f'Best parameters: {model.best_params_}')

# Generate forecasts
forecasts = model.predict(fh=FORECASTING_HORIZON, X=X_test)
forecasts = forecasts.rename_axis('Date')

# Save the reference forecasts in CSV format
OUTPUT_CSV_2 = 'datasets/reference_forecasts.csv'
forecasts.reset_index().to_csv(OUTPUT_CSV_2, index=False)

# Evaluation
print('\nEvaluation')
evaluate_time_series(
    test=y_test.values,
    forecasts=forecasts.values,
    train=y_train.values,
    seasonal_period=SP
)
evaluate_regression(
    y_test=y_test.values, y_pred=forecasts.values, seed=SEED)

# Model persistence
model_path = 'models/sktime/model'
mlflow_sktime.save_model(sktime_model=model, path=model_path)

# Plot actual values
plot_series(
    y_train['Target'],
    y_test['Target'],
    forecasts['Target'],
    labels=['Train', 'Test', 'Forecasts'],
    markers=['.', '*', '+'],
    colors=['tab:blue', 'tab:red', 'tab:green'],
    title='Actual Values vs Forecasts',
    x_label='Date',
    y_label='Daily logarithmic returns'
)
plt.legend(loc='best')
plt.grid(True)
plt.show()
