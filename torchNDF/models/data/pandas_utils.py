import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.arima_model import ARIMA
import logging


def spotrate_lookup(dataframe: pd.DataFrame, date: datetime.date):
    """ Returns the row in dataframe with index closest to `date`"""
    try:
        sliced_dataframe = dataframe[dataframe.index.year == date.year]
        seq = np.array(abs(sliced_dataframe.index.day - date.day) +
                       30 * (abs((sliced_dataframe.index.month - date.month))))
        idx = seq.argmin()
        res = sliced_dataframe.iloc[idx]
    except KeyError:
        logging.info(f'{date.year} not present in dataframe')
        sliced_dataframe = dataframe[str(dataframe.index.year[-1])]
        res = sliced_dataframe.iloc[-1]
    return res


def augment_with_spot_rate(dataframe: pd.DataFrame, spot_dataframe: pd.DataFrame) -> None:
    """ Augments dataframe (in-place) with spot rate feature. dataframe['END_DATE'] used
    to find most recent spot rate information available for currency `dataframe['CURRENCY']`
    in spot_dataframe """
    spot_col = []
    date, curr = None, None
    for _, row in dataframe.iterrows():
        try:
            date = pd.to_datetime(row['END_DATE']).date()
            spot_rates = spotrate_lookup(spot_dataframe, date)
            if spot_rates.any():
                curr = row['CURRENCY']
                spot_data = spot_rates.drop([c for c in spot_rates.index \
                                             if not str(curr) in c])
                spot_col.append(spot_data.item())

        except ValueError:
            logging.debug(f'Incorrect currency or date request to the spot rate dataframe: {curr}, {date}')
            spot_col.append(0)
    spot_data = pd.Series(spot_col)
    spot_data.index = dataframe.index
    dataframe['SPOT_RATE'] = spot_data


def mu_law_encode(signal: pd.DataFrame, quantization_channels: int, norm: bool = True):
    """Mu-law encode"""
    mu = quantization_channels - 1
    if norm: signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1
    magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    signal = np.sign(signal) * magnitude
    signal = (signal + 1) / 2 * mu + 0.5
    quantized_signal = signal.astype(np.int16)
    return quantized_signal


def as_dateindex_filled(dataframe: pd.DataFrame):
    """Fills out a DataFrame adding rows start and end index (daily frequency)"""
    original_columns = dataframe.columns
    dataframe_new = pd.DataFrame({'date': pd.date_range(dataframe.index.min(),
                                                        dataframe.index.max(), freq='D')})
    dataframe_new.index = dataframe_new['date']
    dataframe_new.drop(columns=['date'], inplace=True)
    for col in dataframe:
        dataframe_new[col] = 0.
    dataframe = dataframe_new.join(dataframe, lsuffix='', rsuffix='_ff')
    dataframe.drop(columns=original_columns, inplace=True)
    dataframe.ffill(inplace=True)
    return dataframe


def n_step_returns(dataframe_no_weekends: pd.DataFrame, dataframe_weekends: pd.DataFrame, steps:int):
    """Computes `steps` day returns for each row in `dataframe_no_weekends` using data from
    `dataframe_weekends"""
    returns = []
    for date, _ in dataframe_no_weekends.iterrows():
        start_price = dataframe_no_weekends.loc[date]
        date = date + datetime.timedelta(days=steps)
        if date < dataframe_weekends.index.max():
            end_price = dataframe_weekends.loc[date.date()]
            returns.append((end_price - start_price)/start_price)
    return np.array(returns)


def arima_forecasts(series: pd.Series, split: float = 0.5):
    """1-step ARIMA forecast for `series`"""
    train_idx = int(len(series) * split)
    train, test = series[:train_idx], series[train_idx:]
    history = list(train)
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(4, 1, 0))
        model_fit = model.fit(disp=0)
        pred = model_fit.forecast()[0]
        predictions.append(pred)
        obs = test[t]
        history.append(obs)
    # look-ahead by 1 with full history
    model = ARIMA(history, order=(4, 1, 0))
    model_fit = model.fit(disp=0)
    pred = model_fit.forecast()[0]
    predictions.append(pred)
    # skip first train datapoint (shift right by 1)
    arima_feature = np.concatenate((train[1:], np.array(predictions).flatten()))
    return arima_feature


def augment_with_arima_features(dataframe: pd.DataFrame, trval_split: float = 0.5, column_idx: int = 7) -> None:
    """Augments `dataframe` with ARIMA 1-step forecast features for columns up to `column_idx`"""
    static_columns = dataframe.columns
    for column in static_columns[:column_idx]:
        arima_feature = arima_forecasts(dataframe[column].values, split=trval_split)
        dataframe[f'ARIMA_{column}'] = arima_feature


def augment_with_technical_indicators(dataframe: pd.DataFrame, column_idx: int = 7):
    for col in dataframe.columns[:column_idx]:
        dataframe[f'{col}_ma7'] = dataframe[col].rolling(window=7).mean()
        dataframe[f'{col}_ma21'] = dataframe[col].rolling(window=21).mean()
        dataframe[f'{col}_26ema'] = dataframe[col].ewm(span=26).mean()
        dataframe[f'{col}_12ema'] = dataframe[col].ewm(span=12).mean()
        dataframe[f'{col}_MACD'] = (dataframe[f'{col}_12ema'] - dataframe[f'{col}_26ema'])
        dataframe[f'{col}_20sd'] = dataframe[col].rolling(window=20).std()
        dataframe[f'{col}_upper_band'] = dataframe[f'{col}_ma21'] + (dataframe[f'{col}_20sd'] * 2)
        dataframe[f'{col}_lower_band'] = dataframe[f'{col}_ma21'] - (dataframe[f'{col}_20sd'] * 2)

