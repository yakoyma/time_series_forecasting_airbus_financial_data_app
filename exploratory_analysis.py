"""
===============================================================================
Time Series Project: Airbus Financial Data Forecasting Application
===============================================================================

This file is organised as follows:
1. Load and explore the raw dataset
2. Cleanse the raw dataset
3. Statistical tests
"""
# Standard library
import platform

# Other libraries
import numpy as np
import pandas as pd
import yfinance as yf
import sktime
import sweetviz as sv
import ydata_profiling
import statsmodels


from yfinance import download
from sktime.utils.plotting import plot_series
from sweetviz import analyze
from ydata_profiling import ProfileReport
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Yfinance: {}'.format(yf.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Statsmodels: {}'.format(statsmodels.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load and explore the raw dataset
===============================================================================
"""
print(f'\n\n\n1. Load and explore the raw dataset')


# Load the raw dataset
raw_dataset = download(
    tickers='AIR.PA',
    period='10y',
    interval='1d',
    multi_level_index=False
)

# Display the raw dataset information and description
dataset_info_description(dataset=raw_dataset, max_rows=150)


# Generate the raw dataset report
raw_dataset_report_sv = analyze(source=raw_dataset)
raw_dataset_report_sv.show_html('datasets/raw_dataset_report_sv.html')
raw_dataset_report_ydp = ProfileReport(
    df=raw_dataset, title='Raw Dataset Report')
raw_dataset_report_ydp.to_file('datasets/raw_dataset_report_ydp.html')


# Save the raw dataset in CSV format
OUTPUT_CSV_1 = 'datasets/raw_airbus_financial_data.csv'
raw_dataset.reset_index().to_csv(OUTPUT_CSV_1, index=False)



"""
===============================================================================
2. Cleanse the raw dataset
===============================================================================
"""
print(f'\n\n\n2. Cleanse the raw dataset')


# Convert features to numerical values
dataset = raw_dataset.reset_index()
features = list(dataset.columns)
features.remove('Date')
for feature in features:
    dataset[feature] = pd.to_numeric(dataset[feature], errors='coerce')

    
# Calculate the daily logarithmic returns (the target)
dataset['Target'] = np.log(dataset['Close'] / dataset['Close'].shift(1))


# Calculate the annualised volatility
volatility = dataset['Target'].std() * np.sqrt(252) * 100
print('Annualised Volatility: {:.1f}%'.format(volatility))


# Management of missing data
if dataset.isna().any().any() == True:
    dataset = dataset.dropna()
    dataset.reset_index(inplace=True, drop=True)

# Display dataset information and description
dataset_info_description(dataset=dataset, max_rows=150)


# Generate the dataset report
dataset_report_sv = analyze(source=dataset)
dataset_report_sv.show_html('datasets/dataset_report_sv.html')
dataset_report_ydp = ProfileReport(df=dataset, title='Dataset Report')
dataset_report_ydp.to_file('datasets/dataset_report_ydp.html')


# Save the dataset in CSV format
OUTPUT_CSV_2 = 'datasets/airbus_financial_data.csv'
dataset.to_csv(OUTPUT_CSV, index=False)


# Plot prices
dataset = dataset.set_index('Date')
dataset.index = pd.PeriodIndex(data=dataset.index, freq='B')

plot_series(
    dataset['Open'],
    dataset['Close'],
    dataset['High'],
    dataset['Low'],
    labels=['Open', 'Close', 'High', 'Low'],
    markers=['.', '*', '+', 'o'],
    colors=['tab:blue', 'tab:red', 'tab:green', 'tab:purple'],
    title='Airbus Stock Price',
    x_label='Date',
    y_label='Price'
)
plt.legend(loc='best')
plt.grid(True)
plt.show()


# Plot the target
plot_series(
    dataset['Target'],
    labels=['Target'],
    title='Daily logarithmic returns of Airbus financial data',
    x_label='Date',
    y_label='Daily logarithmic returns'
)
plt.legend(loc='best')
plt.grid(True)
plt.show()



"""
===============================================================================
3. Statistical tests
===============================================================================
"""
print(f'\n\n\n3. Statistical tests')


def get_statistical_tests_result(X, n_lags: int):
    """This function runs the ADF and KPSS tests, displays their results, 
    and plots the ACF and PACF curves.

    Args:
        X (pd.DataFrame): the time series to test
        n_lags (int): the number of lags to use in the tests
    """
    
    adfuller_test_result = adfuller(x=X, maxlag=n_lags, result_object=True)
    print(f'\nAugmented Dickey Fuller (ADF) test result: '
          f'{adfuller_test_result}')
    kpss_test_result = kpss(x=X, nlags=n_lags, result_object=True)
    print(f'\nKwiatkowski-Phillips-Schmidt-Shin (KPSS) test result: '
          f'{kpss_test_result}')
    
    kpss_pvalue = kpss_test_result.pvalue
    adfuller_pvalue = adfuller_test_result.pvalue
    if kpss_pvalue < 0.05 and adfuller_pvalue >= 0.05:
        print('The series is not stationary.')
    elif kpss_pvalue >= 0.05 and adfuller_pvalue < 0.05:
        print('The series is stationary.')
    elif kpss_pvalue >= 0.05 and adfuller_pvalue >= 0.05:
        print('The series is trend stationary.')
    elif kpss_pvalue < 0.05 and adfuller_pvalue < 0.05:
        print('The series is difference stationary.')
    
    # Plot autocorrelation and partial autocorrelation
    plt.figure(figsize=(12, 6))
    plt.plot(
        acf(x=X, nlags=n_lags),
        label='Autocorrelation',
        color='tab:blue'
    )
    plt.plot(
        pacf(x=X, nlags=n_lags),
        label='Partial Autocorrelation',
        color='tab:red'
     )
    plt.title('Autocorrelation and Partial Autocorrelation')
    plt.xlabel('Lags')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()


# Determine whether the Close price is stationary
print('\n\nDetermine whether the Close price is stationary')
get_statistical_tests_result(X=dataset['Close'].values, n_lags=5)

# Detrend the Close price using first-order differentiation
dataset['Close - Diff'] = dataset['Close'].diff(periods=1)
dataset = dataset.dropna()
dataset.reset_index(inplace=True, drop=True)

# Determine whether the detrended Close price is stationary
get_statistical_tests_result(X=dataset['Close - Diff'].values, n_lags=5)

# Plot the detrended Close price
plot_series(
    dataset['Close - Diff'],
    title='Airbus Detrended Close Price',
    x_label='Date',
    y_label='Price'
)
plt.legend(loc='best')
plt.grid(True)
plt.show()


# Determine whether the target is stationary
print('\n\nDetermine whether the target is stationary')
get_statistical_tests_result(X=dataset['Target'].values, n_lags=5)
