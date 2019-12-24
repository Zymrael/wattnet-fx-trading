import argparse
from datetime import datetime, timedelta
import errno
import logging
import os
from pathlib import Path
import re
import sys
import time

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error
import pandas as pd
import pickle
import pytz

log = logging.getLogger(__name__)
format_str = '%(asctime)s | %(name)-10s | %(funcName)s | %(message)s'
logging.Formatter(format_str)
logging.basicConfig(level=logging.INFO)

class OandaRunner():
    '''Historical data fetcher for Oanda FX data. Connects to Oanda API and updates a data
    cache files in _data/oanda/.

    Args:
        instruments (list): instruments to fetch (strings containing base currency and
            quote currency delimited by a underscore).
        dt_from (str): start date (format: "YYYY-MM-DD").
        dt_to (str): start date (format: "YYYY-MM-DD"). Defaults to None (fetch
            data up to the most recent rounded granularity).
        fields (str): one of ['M', 'B', 'A', 'BA', 'MBA'] ([M]id, [B]id, [A]sk,
            Bid and Ask ('BA'), or Mid, Bid and Ask ('MBA').
        granularity (str): data granularity ([S]econd, [M]inute, [H]our, [D]ay, [W]eek or
            [MO]nth).
        frequency (int): data frequency (e.g. 1 if one minute).
        timeout (int): seconds for API request timeout
        n_bars = number of requests (not needed if start and end date provided).
            # max 5000, default 500.
        keep_updated (bool): whether to fetch history once or keep updating.
        update_freq (str): update frequency string (e.g. '1T' is one minute, '1H' one hour,
            etc.).
        data_path (str): data folder name (will be in precog/ root).
    '''
    def __init__(
        self,
        instruments='EUR_USD',
        dt_from='2019-05-15',
        dt_to=None,
        fields='BA',
        granularity='M',
        frequency=1,
        timeout=10,
        n_bars=5000,
        keep_updated=True,
        data_path='_data',
    ):
        self.data_path = Path(data_path)

        if not os.path.exists(data_path):
            log.info(f'creating data path: {data_path}')
            os.makedirs(data_path)

        self.instruments = instruments
        if len(dt_from) != 10:
            raise ValueError('length of date should be 10 (e.g. "2019-05-15")')
        else:
            dt_from = dt_from + 'T00:00:00+0000'

        self.dt_from = self._check_date(dt_from)

        if dt_to:
            dt_to = dt_to + 'T00:00:00+0000'
            self.dt_to = self._check_date(dt_to)
            self.dt_to = self._round_date(
                date=dt_to, granularity=granularity, frequency=frequency)
        else:
            self.dt_to = None
            self.dt_end = self._round_date(
                date=None, granularity=granularity, frequency=frequency, out_string=False)

        self.fields = fields
        self.granularity = granularity
        self.frequency = frequency
        self.timeout = timeout
        self.n_bars = n_bars
        self.keep_updated = keep_updated

        self.check_frequency = 60

    def run(self):
        api = self._start_session(timeout=self.timeout)
        to_check = True
        try:
            if self.instruments:
                while to_check:
                    dt_now = self._round_date(
                        date=None, granularity=self.granularity, frequency=self.frequency,
                        out_string=False)
                    log.info(f'fetching data up to {dt_now}')

                    for i in self.instruments:
                        file_path = self._name_file(date=dt_now, instrument=i)
                        is_file = os.path.isfile(file_path)

                        if not is_file:
                            log.info(f'{i}: data cache not available, will create {file_path}')

                            data = None
                            dt_last = self._round_date(
                                date=self.dt_from, granularity=self.granularity,
                                frequency=self.frequency, out_string=False)

                            while dt_last < dt_now:
                                data = self._append_data(
                                    data=data, api=api, instrument=i, dt_from=dt_last,
                                    dt_to=dt_now)
                                #dt_last = pd.to_datetime(data.index[-1]).tz_localize(pytz.utc)
                                dt_last = pd.to_datetime(data.index[-1]).tz_convert(pytz.utc)
                            file_path = self._name_file(date=self.dt_end, instrument=i)
                            self._save_data(data=data, file_path=file_path)
                        else:
                            log.info(f'{i}: data cache available, loading {file_path}')

                            data = self._load_data(file_path=file_path)
                            dt_last = self._round_date(
                                date=data.index[-1], granularity=self.granularity,
                                frequency=self.frequency, out_string=False)

                            is_update_time = self._check_update(
                                from_time=dt_last, to_time=dt_now)

                            if is_update_time:
                                data = self._append_data(
                                    data=data, api=api, instrument=i, dt_from=dt_last,
                                    dt_to=dt_now)
                                self._save_data(data=data, file_path=file_path)

                    if self.keep_updated:
                        log.info(f'next data check in {self.check_frequency} seconds')
                        time.sleep(self.check_frequency)

                    else:
                        to_check = False
                        log.info('data fetching complete')
                        sys.exit()

            else:
                raise ValueError('no instrument provided')

        except ConnectionError as e:
            log.error(f"ConnectionError: {e}")
            time.sleep(3)

        except V20Error as v20e:
            log.info(f"ERROR {v20e.code}: {v20e.msg}")

        except ValueError as e:
            log.info(f"{e}")

        except Exception as e:
            log.info(f"Unkown error: {e}")

    def _append_data(self, data, api, dt_from, instrument, dt_to=None):
        log.info(f'{instrument}: requesting historical bars from {dt_from}')

        new_data = self._request_data(api=api, instrument=instrument, dt_from=dt_from)
        new_data = self._format_data(data=new_data)

        if dt_to:
            # subset to avoid downloading more data for some instruments (if we passet the
            # granularity/frequency)
            #new_last = pd.to_datetime(new_data.index[-1]).tz_localize(pytz.utc)
            new_last = pd.to_datetime(new_data.index[-1]).tz_convert(pytz.utc)
            if new_last >= dt_to:
                end_idx = new_data.index.get_loc(dt_to.strftime("%Y-%m-%dT%H:%M:%S.%f000Z"))
                new_data = new_data.iloc[:end_idx + 1]

        data = pd.concat([data, new_data])
        log.info(f'new data tail: {data.tail()}')
        log.info(f'data fetched up to {data.index[-1]}')

        return data

    def _check_update(self, from_time, to_time):
        if self.granularity == 'M':
            from_check = from_time.minute
            to_check = to_time.minute

        elif self.granularity == 'H':
            from_check = from_time.hour
            to_check = to_time.hour

        else:
            raise NotImplementedError(f'granularity {self.granularity} not supported')

        out = to_check != from_check

        return out

    def _name_file(self, date, instrument):
        date = date.strftime('%Y%m%d')
        file_name = '_'.join([
            instrument, date, self.fields, str(self.frequency) + self.granularity + '.pickle'])

        file_path = self.data_path / file_name

        return file_path

    def _request_data(self, api, instrument, dt_from):
        if not isinstance(dt_from, str):
             dt_from = dt_from.strftime("%Y-%m-%dT%H:%M:%S+0000")

        if self.granularity == 'MO':
            granularity = 'M'

        else:
            granularity = self.granularity + str(self.frequency)

        self.params = self._parametrize(
            fields=self.fields,
            granularity=granularity,
            n_bars=self.n_bars,
            dt_from=dt_from,
            dt_to=self.dt_to,
        )

        req = instruments.InstrumentsCandles(instrument=instrument, params=self.params)
        out = api.request(req)

        return out

    def _format_data(self, data):
        col_from = ['bid.o', 'ask.o', 'bid.h', 'ask.h', 'bid.l', 'ask.l', 'bid.c', 'ask.c']
        col_to = [
            'Open Bid', 'Open Ask', 'High Bid', 'High Ask', 'Low Bid', 'Low Ask',
            'Close Bid', 'Close Ask'
        ]
        data = pd.io.json.json_normalize(data['candles'])
        data = data.set_index('time')
        data = data.loc[:, col_from]
        data.columns = col_to

        return data

    def _check_date(self, date):
        date_format = r'(\d\d\d\d[-]\d\d[-]\d\dT\d\d[:]\d\d[:]\d\d[+]\d\d\d\d)'

        if date:
            correct_format = re.match(date_format, date)

            if not correct_format:
                raise ValueError(
                    f'incorrect date format (require: "YYYY-MM-DDTHH:MM:SS+0000"): {date}')

        else:
            raise ValueError('date not provided')

        return date

    def _start_session(self, timeout):
        token = 'INSERT YOUR TOKEN HERE'

        api_params = {}
        api_params['timeout'] = timeout

        api = API(
            access_token=token,
            environment="practice",
            request_params=api_params,
        )

        return api

    def _parametrize(self, fields, granularity, n_bars, dt_from, dt_to):
        req_params = {}
        req_params["granularity"] = granularity
        req_params["from"] = dt_from
        req_params["price"] = fields

        if n_bars:
            req_params["count"] = n_bars

        if dt_to:
            req_params["to"] = dt_to

        return req_params

    def _round_date(self, date, granularity, frequency, out_string=True):
        if date:
            #date = pd.to_datetime(date).tz_localize(pytz.utc)
            date = pd.to_datetime(date).tz_convert(pytz.utc)

        else:
            date = datetime.now(pytz.utc)

        if granularity == 'M':
            to_round = date.minute % frequency
            dt_excess = timedelta(
                hours=0, minutes=to_round, seconds=date.second,
                microseconds=date.microsecond)


        elif granularity == 'H':
            to_round = date.hour % frequency
            dt_excess = timedelta(
                hours=to_round, minutes=date.minute, seconds=date.second,
                microseconds=date.microsecond)

        else:
            raise NotImplementedError(f'rounding not implemented for {granularity} granularity')

        dt_round = date - dt_excess
        if out_string:
            out = dt_round.strftime("%Y-%m-%dT%H:%M:%S+0000")

        else:
            out = dt_round

        return out

    def _save_data(self, data, file_path):
        '''Save retrieved data to local.
        '''
        with open(file_path, "wb") as output_file:
            pickle.dump(data, output_file)
        log.info(f'data cached in {file_path}')
        log.info(f'new tail of data:\n{data.tail()}')

    def _load_data(self, file_path):
        '''Load cached data, if any.
        '''
        if not os.path.isfile(file_path):
            raise IOError(f'no such file: {file_path}')

        with open(file_path, "rb") as input_file:
            log.info(f'data loaded from {file_path}')
            out = pickle.load(input_file)

            return out

    def _remove_data(self, file_path):
        '''Remove data, if any.
        '''
        try:
            os.remove(file_path)
            log.info(f'data removed from {file_path}')

        except OSError as e:
            # errno.ENOENT is "no such file or directory"
            if e.errno != errno.ENOENT:
                raise


if __name__ == '__main__':
    example_text = '''Examples of use:
    python scripts/oanda_runner.py
        --ids EUR_USD USD_JPY
        --start 2019-05-19f
        --fields BA
        --g M
        --frequency 1
        --live
    '''

    parser = argparse.ArgumentParser(
        description='Oanda data fetcher.',
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--ids',
        type=str,
        dest='instruments',
        nargs='+',
        help='specify instruments to fetch')

    parser.add_argument(
        '--start',
        type=str,
        dest='from_dt',
        help='start date')

    parser.add_argument(
        '--end',
        type=str,
        dest='to_dt',
        default=None,
        help='end date')

    parser.add_argument(
        '--fields',
        type=str,
        dest='fields',
        default='BA',
        choices=['B', 'A', 'BA', 'MBA'],
        help='fields requested (mid, bid, ask or their combination)')

    parser.add_argument(
        '--granularity',
        type=str,
        dest='granularity',
        choices=['S', 'M', 'H', 'D', 'W', 'CRITICAL'],
        help='data granularity')

    parser.add_argument(
        '--frequency',
        type=int,
        dest='frequency',
        help='data frequency')

    parser.add_argument(
        '--path',
        type=str,
        dest='data_path',
        default='_data',
        help='path to data folder from script run path')

    parser.add_argument(
        '--live',
        dest='keep_updated',
        action='store_true',
        help='whether to keep data updated at each granularity/frequency interval')

    args = parser.parse_args()

    oanda = OandaRunner(
        instruments=args.instruments,
        dt_from=args.from_dt,
        dt_to=args.to_dt,
        fields=args.fields,
        granularity=args.granularity,
        frequency=args.frequency,
        keep_updated=args.keep_updated,
        data_path=args.data_path,
    )

    try:
        oanda.run()
    except (KeyboardInterrupt, SystemExit):
        log.info('Exit on KeyboardInterrupt or SystemExit')
        sys.exit()