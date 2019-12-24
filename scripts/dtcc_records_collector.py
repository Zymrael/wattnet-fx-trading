""" Script to collect and preprocess DTCC NDF records"""

import argparse
import logging
import logging.config
from pathlib import Path
import pandas as pd
import features
from torchNDF.script_utils import *
from torchNDF.data.pandas_utils import *

def process_chunk(chunk_df, spot_df):
    """Preprocess a DataFrame chunk"""
    timestamp_features = features.NDF_TIMESTAMP_COLUMNS_DTCC

    logging.info('Dropping rows with duplicate DISSEMINATION_ID')
    chunk_df = chunk_df[~chunk_df.index.duplicated(keep='first')]

    chunk_df.fillna(0, inplace=True)

    usd_is_1 = chunk_df['NOTIONAL_CURRENCY_1'] == 'USD'
    usd_is_2 = chunk_df['NOTIONAL_CURRENCY_2'] == 'USD'

    logging.debug('numrows before usd drop: %d', len(chunk_df))
    chunk_df = chunk_df[usd_is_1 | usd_is_2]
    logging.debug('numrows after usd drop: %d', len(chunk_df))

    logging.info('Dropping currencies not being considered')
    relevant_ccy_1 = chunk_df['NOTIONAL_CURRENCY_1'].isin(features.NDF_CURRENCIES_DTCC)
    relevant_ccy_2 = chunk_df['NOTIONAL_CURRENCY_2'].isin(features.NDF_CURRENCIES_DTCC)
    chunk_df = chunk_df[relevant_ccy_1 & relevant_ccy_2]
    logging.debug('numrows after curr mask: %d', len(chunk_df))

    logging.info('Adding pluses bool feature')
    augment_with_pluses(chunk_df, usd_is_1, usd_is_2)
    logging.debug('numrows after plus feat: %d', len(chunk_df))

    logging.info('Setting correct types for various features...')
    for feature in features.NDF_BOOL_COLUMNS_DTCC:
        chunk_df[feature].astype('bool', inplace=True)
    for feature in features.NDF_CONT_COLUMNS_DTCC:
        chunk_df[feature] = to_int(chunk_df[feature])
    for feature in features.NDF_CATEGORY_COLUMNS_DTCC:
        chunk_df[feature].astype('category', inplace=True)
    for feature in features.NDF_DATE_COLUMNS_DTCC:
        chunk_df[feature] = pd.to_datetime(chunk_df[feature], errors='coerce')
    for feature in features.NDF_TIMESTAMP_COLUMNS_DTCC:
        chunk_df[feature] = pd.to_datetime(chunk_df[feature], errors='coerce')

    logging.info('Adding ndf rate')
    amounts_to_ndf_rate(chunk_df, usd_is_1, usd_is_2)
    logging.debug('numrows after ndf rate: %d', len(chunk_df))

    logging.info('Removing outdated trade information based on DISSEMINATION_IDs...')
    chunk_df = remove_dissemination_id_changes(chunk_df)
    logging.debug('numrows after diss_id removal: %d', len(chunk_df))

    logging.info('Adding spot rate feature')
    augment_with_spot_rate(chunk_df, spot_df)
    logging.debug('numrows after spot rate: %d', len(chunk_df))

    logging.info('Adding term length feature')
    augment_with_delta(chunk_df, 'END_DATE', 'EFFECTIVE_DATE', 'TERM_LENGTH')

    logging.info('Splitting timestamp columns into date and time features...')
    for timestamp in timestamp_features:
        split_timestamp(chunk_df, timestamp)
    logging.debug('numrows after timestamp split: %d', len(chunk_df))


    cols_out = ['INDICATION_OF_OTHER_PRICE_AFFECTING_TERM', 'BLOCK_TRADES_AND_LARGE_NOTIONAL_OFF-FACILITY_SWAPS']
    cols_in = ['PRICE_AFFECTING_TERM', 'OFF_FACILITY_SWAPS']
    chunk_df = chunk_df.rename(index=str, columns=dict(zip(cols_out, cols_in)))

    chunk_df = chunk_df.drop(columns=['ORIGINAL_DISSEMINATION_ID', 'ACTION',
                                      'ROUNDED_NOTIONAL_AMOUNT_1', 'ROUNDED_NOTIONAL_AMOUNT_2',
                                      'NOTIONAL_CURRENCY_1', 'NOTIONAL_CURRENCY_2',
                                      'SETTLEMENT_CURRENCY'
                                      ]
                             )

    return chunk_df


def main(path, name, mode, chunksize):

    format_str = '%(asctime)s | %(name)-10s | %(funcName)s | %(message)s'
    logging.Formatter(format_str)
    logging.basicConfig(level=logging.INFO, format=format_str)

    save_path = Path('pickle_data')
    if not save_path.exists():
        save_path.mkdir()

    columns = features.NDF_COLUMNS_DTCC

    columns_spot = features.COLUMNS_SPOT
    spot_dataframe = pd.read_csv(f'{path}/fx.csv', usecols=columns_spot,
                          infer_datetime_format=True, index_col=columns_spot[0])
    spot_dataframe.index = pd.to_datetime(spot_dataframe.index)

    if mode=='m':
        logging.info('Reading and merging CSV files...')
        ### TO DO ###
        dataframe = utils.merge_from_folder(path, columns)
        dataframe = process_chunk(dataframe, spot_dataframe)

        logging.info('Saving processed pd.DataFrame...')
        dataframe.to_pickle('pickle_data/all_data_processed.pickle')

    if mode== 'r':
        logging.info('Chunking big csv file...')
        count = 1

        n_rows = sum(1 for row in open(f'{path}/{name}.csv', 'r'))
        logging.info('There are %d rows in the file', n_rows)

        for chunk_df in pd.read_csv('{}/{}.csv'.format(path, name), index_col=columns[0],
                                 parse_dates=True, infer_datetime_format=True, usecols=columns,
                                 chunksize=chunksize):

            logging.info('Processing chunk: %d', count)
            chunk_df = process_chunk(chunk_df, spot_dataframe)

            logging.info('Saving processed pd.DataFrame...')
            chunk_df.to_pickle(f'pickle_data/slice_{count}.pickle')

            logging.info('Progress: %f', round(100*(round(count*chunksize)/n_rows), 4))
            count += 1

    logging.info('Preprocessing complete')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Load all CSV files from folder and merges them into a pd.DataFrame')

    parser.add_argument('--path', type=str, default='.',
                        help='Path containing CSV files to merge')

    parser.add_argument('--name', type=str, default='all_data',
                        help='Name of single BIG csv file')

    parser.add_argument('--mode', type=str, default='r',
                        choices = ['r', 'm'],
                        help='data script mode: *r* for read (single BIG file) *m* for merge and read')

    parser.add_argument('--chunksize', type=int, default=1000000,
                        help='Size (rows) of each csv chunk. Limit to 10**6 to avoid memory issues')
    return parser


def parse_run():
    parser = get_args_parser()
    args = parser.parse_args()
    main(**vars(args))


if __name__ == '__main__':
    parse_run()