"""Utils exclusively utilized by data collection and preprocessing scripts"""

import re
import pandas as pd

def remove_dissemination_id_changes(dataframe:pd.Dataframe):
    """Drops rows in pandas.DataFrame with updated DISSEMINATION_ID information"""
    n_corrections = len(dataframe[dataframe['ACTION'] == 'CORRECT'])
    n_cancels = len(dataframe[dataframe['ACTION'] == 'CANCEL'])
    to_drop = []
    print(f'There have been {n_cancels} cancels and '
          f'{n_corrections} corrections in dissemination IDs')
    for row_idx, row in dataframe.iterrows():
        if row['ACTION'] in ['CORRECT', 'CANCEL']:
            o_id = row['ORIGINAL_DISSEMINATION_ID']
            o_id = int(o_id)
            if o_id in dataframe.index:
                to_drop.append(o_id)
    if len(to_drop) > 0:
        dataframe = dataframe.drop(to_drop, axis=0)
    return dataframe

def to_int(series:pd.Series):
    """Transform values in pandas.Series into a valid format for int conversion
    NaN values are replaced by 0. Removes [,.+]. Trailing decimals are removed."""
    series.fillna(0, inplace=True)
    series = series.astype(str).apply(lambda x:
                             re.sub(r'[.]+\d+$', '', x))
    series = series.astype(str).str.replace(r'[,.+]', '')
    series = series.astype(int)
    return series

def augment_with_pluses(dataframe:pd.Dataframe, usd_is_1:pd.Series, usd_is_2:pd.Series):
    """Augment DataFrame with bool feature flagging whether currency amount strings contain '+'"""
    find_plus = lambda elem: str(elem).find('+')
    plus_1 = dataframe['ROUNDED_NOTIONAL_AMOUNT_1'].astype(str).apply(find_plus) != -1
    plus_2 = dataframe['ROUNDED_NOTIONAL_AMOUNT_2'].astype(str).apply(find_plus) != -1
    dataframe.loc[:, 'PLUS_USD'] = (usd_is_1 & plus_1) | (usd_is_2 & plus_2)
    dataframe.loc[:, 'PLUS_CCY'] = (usd_is_2 & plus_1) | (usd_is_1 & plus_2)


def amounts_to_ndf_rate(dataframe:pd.Dataframe, usd_is_1:pd.Series, usd_is_2:pd.Series) -> None:
    """Computes NDF rates from notional amounts and augments `dataframe` with an NDF rate column"""
    dataframe.loc[usd_is_1, 'CURRENCY'] = dataframe[usd_is_1]['NOTIONAL_CURRENCY_2']
    dataframe.loc[usd_is_2, 'CURRENCY'] = dataframe[usd_is_2]['NOTIONAL_CURRENCY_1']

    dataframe.loc[usd_is_1, 'USD_AMOUNT'] = dataframe['ROUNDED_NOTIONAL_AMOUNT_1']
    dataframe.loc[usd_is_2, 'USD_AMOUNT'] = dataframe['ROUNDED_NOTIONAL_AMOUNT_2']
    dataframe.loc[usd_is_2, 'CCY_AMOUNT'] = dataframe['ROUNDED_NOTIONAL_AMOUNT_1']
    dataframe.loc[usd_is_1, 'CCY_AMOUNT'] = dataframe['ROUNDED_NOTIONAL_AMOUNT_2']

    dataframe.loc[:, 'NDF_RATE'] = dataframe['CCY_AMOUNT'] / dataframe['USD_AMOUNT']

def split_timestamp(dataframe:pd.Dataframe, colname:str) -> None:
    """Splits timestamp pd.Series into a time feature and a date feature."""
    dataframe[colname] = pd.to_datetime(dataframe[colname])
    date, time = dataframe[colname].dt.date, dataframe[colname].dt.time
    dataframe[f'{colname}_TIME'] = pd.to_datetime(time)
    dataframe[f'{colname}_DATE'] = date


def augment_with_delta(dataframe:pd.Dataframe, feature_1:str, feature_2:str, new_name:str) -> None:
    """ Augments pandas.DataFrame with column counting number of days between two
    datetime.date features

    Args:
        dataframe (pandas.DataFrame)
        feature_1 (str): start date feature for difference calculation
        feature_2 (str): end date feature for difference calculation
        new_name (str): new feature name
    """
    delta = dataframe[feature_1] - dataframe[feature_2]
    delta_days = delta.apply(lambda x: x.days)
    dataframe[new_name] = delta_days