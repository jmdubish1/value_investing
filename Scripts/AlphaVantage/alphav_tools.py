from datetime import datetime
import datetime as dt
import os

import numpy as np
import pandas as pd
import alphaVantageAPI
import stat
import sched, time, threading
import json
import twelvedata
import scipy.stats as sstats


def check_files(data_loc, second_loc, symbols, file_age):
    """Checks to see if folder and file exists and if it does, when the last time it was downloaded in order to maximize
     API credits. Returns a list of files to refresh.

    :param : data_loc: str, main data folder
    :param : second_loc: str, secondary folder location for specific data types
    :param : symbols: list or series of ticker symbols to check
    :param : file_age: int, how old the oldest file in hours is allowed to be"""

    if not os.path.exists(f'{data_loc}\\{second_loc}'):
        os.mkdir(f'{data_loc}\\{second_loc}')

    refresh = []
    for symbol in set(symbols):
        if os.path.exists(f'{data_loc}\\{second_loc}\\{symbol}.csv'):
            filestatsobj = os.stat(f'{data_loc}\\{second_loc}\\{symbol}.csv')
            modified_time = time.ctime(filestatsobj[stat.ST_MTIME])
            print('File:          ', str(symbol))
            print('Modified Time: ', modified_time)
            time_dif = datetime.now() - pd.to_datetime(modified_time)
            print('File Age:      ', time_dif)
            if time_dif > dt.timedelta(hours=file_age):
                refresh.append(symbol)
        else:
            refresh.append(symbol)

    return refresh


def clean_symbols(listings):
    """Cleans the list of traded securities to only those on NYSE and NASDAQ.

    :param : df of securities"""

    try:
        if listings['exchange']:
            symbols = listings[(listings['exchange'] == 'NYSE')|(listings['exchange'] == 'NASDAQ')]['symbol']
    except KeyError:
        symbols = listings['Ticker']

    symbols_clean = []
    for symbol in symbols:
        symbols_clean.append(symbol.split("-")[0])

    return symbols_clean


def get_data(data_func, data_type, data_loc, listings, av, lookback):
    """Used to get various datasets from AlphaVantage, then put in their right folder

    :param : data_func: AlphaVantage class method; use: overview, balance, cashflow, income, data"""

    symbols_clean = clean_symbols(listings)
    refresh = check_files(data_loc, data_type, symbols_clean, lookback)

    i = 0
    missing_downloads = []
    print(f'Downloading {data_type}')
    print('Downloading %s Files' % len(refresh))
    completed = 0
    not_found = 0
    while len(refresh) > 0:
        for symbol in refresh:
            print(symbol)
            key_err = 0
            completed += 1
            try:
                fd = data_func(symbol)
                fd = json.loads(fd)

            except ValueError:
                print(str(symbol), ' Not Found')
                missing_downloads.append(symbol)
                refresh.remove(symbol)
                not_found += 1
            except KeyError:
                key_err += 1
                if key_err < 4:
                    print(f'Keyerror: {symbol}, waiting 5 seconds')
                    time.sleep(5)
                    refresh.remove(symbol)

            file_to_save = f'{data_loc}\\{data_type}\\{symbol}.csv'
            if data_type == 'BALANCE_SHEET' or  \
                    data_type == 'CASHFLOW' or \
                    data_type == 'INCOME_STATEMENT':
                data = pd.DataFrame.from_dict(fd['annualReports'])
            elif data_type == 'EARNINGS':
                data = pd.DataFrame.from_dict(fd['annualEarnings'])
            elif data_type == 'TIME_SERIES_DAILY_ADJUSTED':
                data = pd.DataFrame.from_dict(fd['Time Series (Daily)']).T
            else:
                data = pd.DataFrame(fd, index=[0])

            data.to_csv(file_to_save)
            print(f'Data saved to : {file_to_save} : {completed} Completed/{len(refresh)} Remaining')
            i += 1
            refresh.remove(symbol)

            if not av.premium:
                print(f'{int(len(refresh)*15/60)} Minutes Remaining')
            else:
                print(f'{int(len(refresh)*(60/av.premium_max)/60)} Minutes Remaining')

    print('Download Complete')
    print('Missing Symbols: ', missing_downloads)

    miss_df = pd.DataFrame(missing_downloads)
    miss_df.to_csv(f'{data_loc}\\{data_type}\\missing_balance.csv')

    print(missing_downloads, 'Missing')


def request_last_day(sym_batch, twelve):
    ts = twelve.time_series(symbol=sym_batch,
                            interval="1day",
                            outputsize=5000,
                            timezone="America/Chicago")
    return ts.as_pandas()


def get_spx_daily(recent, data_path):
    api_key = 'a437851afba948fcb948922d45526699'
    td = twelvedata.TDClient(apikey=api_key)
    per_batch = 8
    refresh = []
    symbols_clean = ['QQQ', 'SPX']
    for symbol in set(symbols_clean):
        if os.path.exists(data_path + '\\' + symbol + '.csv'):
            filestatsobj = os.stat(data_path + '\\' + symbol + '.csv')
            modified_time = time.ctime(filestatsobj[stat.ST_MTIME])
            print('File:          ', str(symbol))
            print('Modified Time: ', modified_time)
            time_dif = dt.datetime.now() - pd.to_datetime(modified_time)
            print('File Age:      ', time_dif)
            if time_dif > dt.timedelta(hours=recent):
                refresh.append(symbol)
        else:
            refresh.append(symbol)

    print(f'Downloading {len(refresh)} files...')
    for st in range(0, len(refresh), per_batch):

        sym_batch = refresh[0:per_batch]
        print(sym_batch)
        try:
            daily_adj = request_last_day(sym_batch, td)
            daily_adj.index.set_names(["symbol","date"], inplace=True)
            [refresh.remove(x) for x in sym_batch]
            daily_adj.reset_index(inplace=True)
            for sym in sym_batch:
                temp_df = daily_adj[daily_adj.symbol == sym]
                temp_df.rename(columns={'date': '', 'open': '1. open', 'high': '2. high', 'low': '3. low',
                                        'close': '5. adjusted close', 'volume': '6. volume'}, inplace=True)
                print(temp_df)
                temp_df.to_csv(f'{data_path}\\{sym}.csv', index=False)
        except twelvedata.exceptions.BadRequestError as e:
            print(f'Error: {str(e)}')
    print('SPX Downloaded')




