"""
===============================================================================
Time Series Project: Airbus Financial Data Forecasting Application
===============================================================================
"""
# Standard library
import platform

# Other libraries
import numpy as np
import pandas as pd
import yfinance as yf
import sktime
import holidays
import joblib
import evidently
import gradio as gr


from datetime import datetime, timedelta, date
from yfinance import download
from sktime.utils import mlflow_sktime
from holidays import country_holidays, list_supported_countries
from joblib import load
from evidently import Report
from evidently.presets import DataDriftPreset
from gradio import Interface, Textbox, Dropdown, LinePlot


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Yfinance: {}'.format(yf.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('Holidays: {}'.format(holidays.__version__))
print('Joblib: {}'.format(joblib.__version__))
print('Evidently: {}'.format(evidently.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_retraining_decision(user_country_code: str):
    """This function detects data drift, forecasts drift, and reports a
    re-training decision.

    Args:
        user_country_code (str): the country code for the holiday period to
                                 take into account selected by the user

    Returns:
        retraining_decision_dataset (pd.DataFrame): The Pandas Dataframe
                                                    containing the
                                                    retraining decision
    """

    # Paths of datasets
    INPUT_CSV_1 = 'datasets/reference_dataset.csv'
    INPUT_CSV_2 = f'datasets/reference_forecasts.csv'
    INPUT_CSV_3 = 'datasets/current_dataset.csv'
    INPUT_CSV_4 = 'datasets/current_forecasts.csv'

    # Load the datasets
    reference_dataset = pd.read_csv(INPUT_CSV_1)
    reference_forecasts = pd.read_csv(INPUT_CSV_2)
    current_dataset = pd.read_csv(INPUT_CSV_3)
    current_forecasts = pd.read_csv(INPUT_CSV_4)

    # Take the holiday period into account
    reference_dataset = reference_dataset.set_index('Date')
    reference_dataset.index = pd.to_datetime(reference_dataset.index)
    reference_dataset_holidays = country_holidays(
        country=user_country_code,
        years=reference_dataset.index.dropna().year.unique()
    )
    reference_dataset = reference_dataset[
        ~reference_dataset.index.normalize().isin(
        reference_dataset_holidays)
    ]
    reference_forecasts = reference_forecasts.set_index('Date')
    reference_forecasts.index = pd.to_datetime(reference_forecasts.index)
    reference_forecasts_holidays = country_holidays(
        country=user_country_code,
        years=reference_forecasts.index.dropna().year.unique()
    )
    reference_forecasts = reference_forecasts[
        ~reference_forecasts.index.normalize().isin(
        reference_forecasts_holidays)
    ]

    # Data drift analysis
    features = ['HML', 'RVOL']
    reference_feature = reference_dataset[features].reset_index(drop=True)
    current_feature = current_dataset[features].reset_index(drop=True)
    data_report = Report(metrics=[DataDriftPreset(columns=features)])
    data_drift_result = data_report.run(
        reference_data=reference_feature,
        current_data=current_feature
    )
    data_drift_result_dict = data_drift_result.dict()
    data_drift_metrics = data_drift_result_dict['metrics'][0]['value']
    data_drift_share = data_drift_metrics['share']
    data_drift_count = data_drift_metrics['count']
    data_drift = data_drift_count > 0
    print(f'\n\nData drift metrics: {data_drift_metrics}')
    print(f'Data drift share: {data_drift_share}')
    print(f'Data drift result: {data_drift}')

    # Forecasts drift analysis
    forecasts_report = Report(metrics=[DataDriftPreset(columns=['Target'])])
    forecasts_drift_result = forecasts_report.run(
        reference_data=reference_forecasts[['Target']],
        current_data=current_forecasts[['Target']]
    )
    forecasts_drift_result_dict = forecasts_drift_result.dict()
    forecasts_drift_metrics = forecasts_drift_result_dict[
        'metrics'][0]['value']
    forecasts_drift_share = forecasts_drift_metrics['share']
    forecasts_drift_count = forecasts_drift_metrics['count']
    forecasts_drift = forecasts_drift_count > 0
    print(f'\nForecasts drift metrics: {forecasts_drift_metrics}')
    print(f'Forecasts drift share: {forecasts_drift_share}')
    print(f'Forecasts drift result: {forecasts_drift}')

    # Re-training decision result
    if data_drift and forecasts_drift:
        retraining_decision_result = 'Re-training required'
    else:
        retraining_decision_result = 'No need for re-training'
    print(f'\nRe-training decision result: {retraining_decision_result}')

    # Create the re-training decision dataset
    retraining_decision_dataset = pd.DataFrame(
        {
            'Data drift share': [data_drift_share],
            'Data drift': [data_drift],
            'Forecasts drift share': [forecasts_drift_share],
            'Forecasts drift': [forecasts_drift],
            'Re-training decision': [retraining_decision_result]
        }
    )
    print(f'Re-training decision dataset:\n{retraining_decision_dataset}')
    return retraining_decision_dataset


def get_forecasts(user_forecasting_horizon: int, user_country_code: str):
    """This function detects data drift, forecasts drift, and reports a
    re-training decision for Gradio application.

    Args:
        user_forecasting_horizon (int): the period (in business days) into
                                        the future for the forecasts
                                        selected by the user
        user_country_code (str): the country code for the holiday period to
                                 take into account selected by the user

    Returns:
        response (str): the text to be displayed to the user
        forecasts (pd.DataFrame): the Pandas DataFrame containing
                                  the forecasts
    """

    try:

        # Load the reference dataset (test set)
        INPUT_CSV = 'datasets/reference_dataset.csv'
        reference_dataset = pd.read_csv(INPUT_CSV)

        # Download the dataset
        start_date = datetime.strptime(
            reference_dataset['Date'].iloc[0], '%Y-%m-%d') - timedelta(days=1)
        start_date = start_date.strftime('%Y-%m-%d')
        raw_current_dataset = download(
            tickers='AIR.PA',
            start=start_date,
            end=str(date.today()),
            interval='1d',
            multi_level_index=False
        )
        current_dataset = raw_current_dataset.reset_index()

        # Save the raw current dataset in CSV format
        OUTPUT_CSV_1 = 'datasets/raw_current_dataset.csv'
        raw_current_dataset.reset_index().to_csv(OUTPUT_CSV_1, index=False)

        # Feature selection
        X = current_dataset.filter(['Date', 'Close', 'High', 'Low', 'Volume'])

        # Create the target (the daily logarithmic returns)
        X['Target'] = np.log(X['Close'] / X['Close'].shift(1))

        # Management of missing data
        if X.isna().any().any() == True:
            X = X.dropna()
            X.reset_index(inplace=True, drop=True)

        # Create the high minus low indicator (HML) feature
        X['HML'] = X['High'] - X['Low']

        # Create the relative volume indicator (RVOL) feature
        X = X.set_index('Date')
        X['RVOL'] = X['Volume'] / X['Volume'].rolling(
            window=50, min_periods=1).mean()
        X['RVOL'] = X['RVOL'].fillna(0)

        # Take the holiday period into account
        X_holidays = country_holidays(
            country=user_country_code, years=X.index.dropna().year.unique())
        X = X[~X.index.normalize().isin(X_holidays)]

        # Save the current forecasts in CSV format
        OUTPUT_CSV_2 = 'datasets/current_dataset.csv'
        X.reset_index().to_csv(OUTPUT_CSV_2, index=False)

        # Normalisation
        scaler_path = 'models/scaler/scaler.joblib'
        scaler = load(filename=scaler_path)
        X_scaled = pd.DataFrame(
            data=scaler.transform(X[['HML', 'RVOL']]),
            index=X.index,
            columns=['HML', 'RVOL']
        )
        X_scaled.index = pd.PeriodIndex(data=X_scaled.index, freq='B')

        # Load the pre-trained model
        model_path = f'models/sktime/model'
        model = mlflow_sktime.load_model(model_uri=model_path)

        # Generate forecasts
        forecasting_horizon = len(X_scaled) + user_forecasting_horizon + 2
        forecasts = model.predict(fh=forecasting_horizon, X=X_scaled)

        # Save the current forecasts in CSV format
        OUTPUT_CSV_3 = 'datasets/current_forecasts.csv'
        forecasts.reset_index().to_csv(OUTPUT_CSV_3, index=False)

        # Take the holiday period into account
        forecasts.index = forecasts.index.astype(str)
        forecasts.index = pd.to_datetime(forecasts.index)
        forecasts_holidays = country_holidays(
            country=user_country_code,
            years=forecasts.index.dropna().year.unique()
        )
        forecasts = forecasts[~forecasts.index.normalize().isin(
            forecasts_holidays)]

        # Create the final dataset
        forecasts_dataset = forecasts.reset_index()
        forecasts_dataset = forecasts_dataset.rename(
            columns={'index': 'Date'})
        forecasts_dataset['Date'] = pd.to_datetime(
            forecasts_dataset['Date'], format='%Y-%m-%d')

        # Retrieve a subset from the current date
        forecasts_dataset = forecasts_dataset[
            forecasts_dataset['Date'].dt.date >= date.today()]
        end_date = forecasts_dataset['Date'].iloc[-1].strftime('%d-%m-%Y')

        # Calculate the annualised volatility
        volatility = forecasts_dataset['Target'].std() * np.sqrt(252) * 100
        response = (f'Here are the daily logarithmic returns forecasts for '
                    f'Airbus stock data up to {end_date}.'
                    f'\nThe annualised volatility is {round(volatility, 1)}%.')

        # Get re-training decision dataset
        retraining_decision_dataset = get_retraining_decision(
            user_country_code=user_country_code)

        # Save the re-training decision dataset in CSV format
        OUTPUT_CSV_4 = 'datasets/retraining_decision_dataset.csv'
        retraining_decision_dataset.to_csv(OUTPUT_CSV_4, index=False)

    except Exception as error:
        forecasts_dataset = pd.DataFrame()
        response = f'The following error occurred: {error}'
    return response, forecasts_dataset



# Instantiate the app
app = Interface(
    fn=get_forecasts,
    inputs=[
        Dropdown(
            choices=list(range(5, 25, 5)),
            label='Please select the future period (in business days)',
            type='value'
        ),
        Dropdown(
            choices=list(list_supported_countries().keys()),
            label='Please select the country code for the holiday period',
            type='value'
        )
    ],
    outputs=[
        Textbox(label='Forecasts result'),
        LinePlot(
            x='Date',
            y='Target',
            title='Daily logarithmic returns forecasts',
            x_title='Date',
            y_title='Daily logarithmic returns'
        )
    ],
    title='Airbus Financial Data Forecasting Application'
)



if __name__ == '__main__':
    app.launch()
