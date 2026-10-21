"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from csv import Sniffer
from sktime.performance_metrics import forecasting
from sklearn import metrics



def load_dataset(file_path, encoding):
    """This function loads a csv file and finds the type of separators by
     sniffing the file.

    Args:
        file_path (str): the csv file path
        encoding (str): encoding to use for reading and writing

    Returns:
        dataset (pd.DataFrame): the loaded Pandas data set
    """

    with open(file_path, 'r') as csvfile:
        separator = Sniffer().sniff(csvfile.readline()).delimiter
    dataset = pd.read_csv(
        filepath_or_buffer=file_path,
        sep=separator,
        encoding=encoding,
        encoding_errors='ignore',
        on_bad_lines='skip'
    )
    return dataset


def dataset_info_description(dataset, max_rows: int):
    """This function displays the information and description of a Pandas
    DataFrame.

    Args:
        dataset (pd.DataFrame): the Pandas DataFrame
        max_rows (int): the maximum number of rows in the dataset to be
                        displayed
    """

    # Display dimensions of the dataset
    print(f'\nDimensions of the dataset: {dataset.shape}')

    # Display information about the dataset
    print(f'\nInformation about the dataset:\n{dataset.info()}')

    # Display the description of the dataset
    print(f'\nDescription of the dataset:\n{dataset.describe(include="all")}')

    # Display the head and the tail of the dataset
    print(f'\nDescription of the dataset:\n'
          f'{pd.concat([dataset.head(max_rows), dataset.tail(max_rows)])}')


def evaluate_time_series(test, forecasts, train, seasonal_period):
    """This function evaluates Time Series model performance 
    using Darts library.

    Args:
        test (array-like): the test set series
        forecasts (array-like): the series containing the forecasts
        train (array-like): the train set series
        seasonal_period (int): the seasonal period of the Time Series data set
    """
    
    print('\n\nMASE: {:.3f}'.format(forecasting.mean_absolute_scaled_error(
        y_true=test, y_pred=forecasts, y_train=train, sp=seasonal_period)))
    print('MdASE: {:.3f}'.format(forecasting.median_absolute_scaled_error(
        y_true=test, y_pred=forecasts, y_train=train, sp=seasonal_period)))
    print('MSSE: {:.3f}'.format(forecasting.mean_squared_scaled_error(
        y_true=test, y_pred=forecasts, y_train=train, sp=seasonal_period)))
    print('MdSSE: {:.3f}'.format(forecasting.median_squared_scaled_error(
        y_true=test, y_pred=forecasts, y_train=train, sp=seasonal_period)))
    print('MAE: {:.3f}'.format(forecasting.mean_absolute_error(
        y_true=test, y_pred=forecasts)))
    print('MdAE: {:.3f}'.format(forecasting.median_absolute_error(
        y_true=test, y_pred=forecasts)))
    print('MSE: {:.3f}'.format(forecasting.mean_squared_error(
        y_true=test, y_pred=forecasts)))
    print('MdSE: {:.3f}'.format(forecasting.median_squared_error(
        y_true=test, y_pred=forecasts)))
    print('MAPE: {:.3f}'.format(forecasting.mean_absolute_percentage_error(
        y_true=test, y_pred=forecasts)))
    print('MdAPE: {:.3f}'.format(forecasting.median_absolute_percentage_error(
        y_true=test, y_pred=forecasts)))
    print('MSPE: {:.3f}'.format(forecasting.mean_squared_percentage_error(
        y_true=test, y_pred=forecasts)))
    print('MdSPE: {:.3f}'.format(forecasting.median_squared_percentage_error(
        y_true=test, y_pred=forecasts)))
    print('GMAE: {:.3f}'.format(forecasting.geometric_mean_absolute_error(
        y_true=test, y_pred=forecasts)))
    print('GMSE: {:.3f}'.format(forecasting.geometric_mean_squared_error(
        y_true=test, y_pred=forecasts)))


def evaluate_regression(y_test, y_pred, seed):
    """This function evaluates Regression model performance 
    using Scikit-learn library.

    Args:
        y_test (array-like): the test labels
        y_pred (array-like): the predicted labels
        seed (int): the random state value in order to ensure reproducibility
    """

    print('\nMSE: {:.3f}'.format(metrics.mean_squared_error(
        y_true=y_test, y_pred=y_pred)))
    print('MAE: {:.3f}'.format(metrics.mean_absolute_error(
        y_true=y_test, y_pred=y_pred)))
    print('MAPE: {:.3f}'.format(metrics.mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred)))
    print('MdAE: {:.3f}'.format(metrics.median_absolute_error(
        y_true=y_test, y_pred=y_pred)))
    print('Maximum residual error: {:.3f}'.format(metrics.max_error(
        y_true=y_test, y_pred=y_pred)))
    print('Explained variance score: {:.3f}'.format(
        metrics.explained_variance_score(y_true=y_test, y_pred=y_pred)))
    print('R²: {:.3f}'.format(metrics.r2_score(
        y_true=y_test, y_pred=y_pred)))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    metrics.PredictionErrorDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        kind='actual_vs_predicted',
        ax=axs[0],
        random_state=seed
    )
    axs[0].set_title('Actual Values vs Predicted Values')
    axs[0].grid(True)
    metrics.PredictionErrorDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        kind='residual_vs_predicted',
        ax=axs[1],
        random_state=seed
    )
    axs[1].set_title('Residuals Values vs Predicted Values')
    axs[1].grid(True)
    fig.suptitle('Plot the results of predictions')
    plt.show()
