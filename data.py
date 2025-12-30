# -*- coding: utf-8 -*-
'''
Data functions
'''

import argparse
import requests
import logging
import yfinance as yf
import pandas as pd
import tempfile
from os import makedirs
from base64 import b64encode
from tenacity import retry, wait_random, stop_after_attempt
from tqdm import tqdm
from platformdirs import user_cache_dir
from bcb import sgs
from constants import *
from logger import logger

MIN_PARTICIPATION_THRESHOLD = 0.95

def _br_value_to_float(text):
    '''
    Convert Brazilian formatted number to float
    '''
    text = text.replace('.', '')
    text = text.replace(',', '.')
    return float(text)

def _fetch_url(url):
    '''
    Fetch content from a URL
    '''
    agente = {'User-Agent': 'Mozilla/5.0'}
    logger.debug(f'Fetching URL: {url}')
    response = requests.get(url, timeout=10, headers=agente)
    if response.status_code == 200:
        return response
    return None

def _fetch_ifix_year(year):
    '''
    Fetch IFIX data for a given year
    '''
    # Encode parameters
    params = f'{{"index":"IFIX","language":"pt-br","year":"{year}"}}'
    params = b64encode(params.encode())
    url = URL_IFIX + str(params.decode())
    content = _fetch_url(url)
    if content is None:
        return None
    # Convert content to json
    content = content.json()
    # Get list of results
    content = content.get('results', None)
    history_list = []
    # Content is a list of days and month values
    for day_dict in content:
        day = day_dict['day']
        for month in range(1, 13):
            value = day_dict['rateValue' + str(month)]
            if value is not None:
                ifix_dict = {
                    DATE: str(year) + '-' + str(month) + '-' + str(day),
                    IFIX: _br_value_to_float(value)
                }
                history_list.append(ifix_dict)
    return history_list

def _download_ifix():
    '''
    Fetch IFIX data for all years
    '''
    current_year = pd.Timestamp.now().year
    all_history = []
    # Fetch IFIX year by year
    for year in range(IFIX_START_YEAR, current_year + 1):
        year_history = _fetch_ifix_year(year)
        if year_history is not None:
            all_history += year_history
    # Convert to DataFrame
    df_ifix = pd.DataFrame(all_history)
    if not df_ifix.empty:
        # Treat date column and save to file
        df_ifix[DATE] = pd.to_datetime(df_ifix[DATE], 
                                       format='%Y-%m-%d')
        df_ifix.set_index(DATE, inplace=True)
        df_ifix.sort_index(inplace=True)
        df_ifix.to_csv(FILE_IFIX)

def get_ifix_series():
    '''
    Get IFIX series from file or try to download it
    '''
    df_ifix = None
    try:
        df_ifix = pd.read_csv(FILE_IFIX, parse_dates=True,
                              index_col=DATE)
    except:
        _download_ifix()
        df_ifix = pd.read_csv(FILE_IFIX, parse_dates=True,
                              index_col=DATE)
    return df_ifix

@retry(
    wait=wait_random(2, 10),
    stop=stop_after_attempt(5)
)
def _fetch_selic(start, end):
    '''
    Fetch SELIC data from BCB
    '''
    # Get daily SELIC from BCB API
    # 4389: Accumulated annual SELIC rate
    return sgs.get({SELIC: 4389}, start=start, end=end)
    
def _download_selic():
    '''
    Fetch SELIC data from BCB
    '''
    year = SELIC_START_YEAR
    current_year = pd.Timestamp.now().year
    df_selic = pd.DataFrame()
    # Fetch SELIC in windows of 10 years (maximum allowed by API)
    while year <= current_year:
        start = f'{year}-01-01'
        end = f'{year + 9}-12-31'
        # Get annual SELIC by day
        df_current = _fetch_selic(start, end)
        year += 10
        df_selic = pd.concat([df_selic, df_current]) # type: ignore
    # Keep just the last value of each month
    df_selic = df_selic.resample('ME').last()
    df_selic[SELIC] = df_selic[SELIC] / 100.0
    # Save to file
    df_selic.to_csv(FILE_SELIC)

def _get_selic_series():
    '''
    Get SELIC series from file or try to download it
    '''
    df_selic = None
    try:
        df_selic = pd.read_csv(FILE_SELIC, parse_dates=True,
                               index_col=DATE)
    except:
        _download_selic()
        df_selic = pd.read_csv(FILE_SELIC, parse_dates=True,
                               index_col=DATE)
    df_selic.index = df_selic.index.to_period('M') # type: ignore
    return df_selic

def get_selic(year, month):
    '''
    Get SELIC rate for a given month
    '''
    df_selic = _get_selic_series()
    selic_rate = 0.0
    try:
        selic_rate = df_selic.loc[str(year) + '-' + str(month)][SELIC]
    except:
        pass
    return selic_rate

def get_reit_list(only_valid=False):
    '''
    Get the list of REITs from file
    '''
    # All REITs
    filename = FILE_REITS
    if only_valid:
        # Valid REITs
        filename = FILE_REITS_VALID
    with open(filename, 'r') as f:
        reit_list = [line.strip()
                     for line in f.readlines()]
    return reit_list

def _save_valid_reit_list(reit_list):
    '''
    Save the valid list of REITs
    '''
    with open(FILE_REITS_VALID, 'w') as f:
        for reit in reit_list:
            f.write(reit + '\n')

def _get_cache_dir():
    '''
    Get cache directory for the application
    '''
    cache_dir = user_cache_dir(APP_NAME)
    makedirs(cache_dir, exist_ok=True)
    return cache_dir

def _get_filename(ticker, subdir=''):
    '''
    Get the filename for a ticker
    '''
    file_dir = _get_cache_dir()
    if subdir:
        file_dir += '/' + subdir
        makedirs(file_dir,
                 exist_ok=True)
    filename = file_dir + '/' + ticker + '.csv'
    return filename

def _empty_stock_data():
    '''
    Create an empty DataFrame with the required columns
    '''
    columns = [OPEN, HIGH, LOW, CLOSE, VOLUME, DIVIDENDS,
               STOCK_SPLITS]
    df_empty = pd.DataFrame(columns=columns)
    df_empty.index.name = DATE
    return df_empty

@retry(
    wait=wait_random(1, 5),
    stop=stop_after_attempt(3)
)
def _fetch_stock_data(ticker):
    '''
    Fetch stock data for a given ticker
    '''
    logger.debug('Fetching data for %s.', ticker)
    ticker = yf.Ticker(ticker + '.SA')
    df_data = ticker.history(period='max',
                             raise_errors=True)
    # Fix date index
    df_data.reset_index(inplace=True)
    df_data[DATE] = pd.to_datetime(df_data[DATE]).dt.date
    df_data.set_index(DATE, inplace=True)
    if df_data.empty:
        df_data = _empty_stock_data()
    return df_data

def _download_stock_list(ticker_list):
    '''
    Fetch data for a list of tickers
    '''
    # Use tqdm for parallel fetching with progress bar
    for ticker in tqdm(ticker_list, desc='Fetching data'):
        try:
            df_data = _fetch_stock_data(ticker)
            filename = _get_filename(ticker, DIR_STOCK)
            df_data.to_csv(filename)
        except Exception as e:
            logger.error('Error fetching data for %s.', ticker)
            logger.error('%s', e)

def get_stock_data(ticker):
    '''
    Get stock data from cache
    '''
    filename = _get_filename(ticker, DIR_STOCK)
    try:
        return pd.read_csv(filename, parse_dates=True, index_col=DATE)
    except:
        return _empty_stock_data()

def download_all():
    '''
    Download all required data
    '''
    logger.debug('Downloading all IFIX')
    _download_ifix()
    logger.debug('Downloading all SELIC')
    _download_selic()
    logger.debug('Downloading all REIT data')
    reit_list = get_reit_list()
    _download_stock_list(reit_list)

def update_valid_reit_list():
    '''
    Update the valid REIT list based on available data
    '''
    reit_list = get_reit_list(only_valid=False)
    valid_list = []
    for ticker in reit_list:
        df_data = get_stock_data(ticker)
        if df_data is not None and not df_data.empty:
            valid_list.append(ticker)
    logger.debug('Saving valid REIT list. Total %s REITs.', len(valid_list))
    _save_valid_reit_list(valid_list)

def join_history(ticker_list, start_date, end_date, fill_missing=True, 
                 field=CLOSE):
    '''
    Join data for multiple tickers
    '''
    df_joined = pd.DataFrame()
    for ticker in ticker_list:
        df_data = get_stock_data(ticker)
        if df_data is None or df_data.empty:
            continue
        # Filter by date
        df_data = df_data[start_date:end_date]
        # Select only the required field and rename column
        df_data = df_data[[field]]
        df_data.rename(columns={field: ticker}, inplace=True)
        # Join (each ticker as a column)
        df_joined = pd.merge(df_joined, df_data,
                             left_index=True, right_index=True,
                             how='outer')
    # Fill missing values
    if fill_missing:
        df_joined.ffill(inplace=True)
        df_joined.bfill(inplace=True)
    return df_joined

def topk_by_volume(ticker_list, k=0, start_date=None, end_date=None,
                   participation_threshold=MIN_PARTICIPATION_THRESHOLD):
    '''
    Get top k tickers by average volume
    '''
    volume_list = []
    max_days = 0
    for ticker in ticker_list:
        df_data = get_stock_data(ticker)
        if df_data is None or df_data.empty:
            continue
        if start_date is not None and end_date is not None:
            df_data = df_data[start_date:end_date]
        # Consider VOLUME as VOLUME * CLOSE
        df_data[VOLUME] = df_data[VOLUME] * df_data[CLOSE]
        volume_sum = df_data[[VOLUME]].sum().values[0]
        days_count = df_data[df_data[VOLUME] > 0].shape[0]
        if days_count > max_days:
            max_days = days_count
        volume_list.append((ticker, volume_sum, days_count))
        # logger.debug('Ticker %s: volume sum %.2f over %d days',
        #              ticker, volume_sum, days_count)
        df_data = df_data[[VOLUME]]
    # Filter out zero less than 95% of days
    filtered_volume_list = []
    for item in volume_list:
        ticker, volume_sum, days_count = item
        if days_count >= participation_threshold * max_days:
            filtered_volume_list.append((ticker, volume_sum))
    volume_list = filtered_volume_list
    # Sort by volume descending
    volume_list.sort(key=lambda x: x[1], reverse=True)
    if k > 0:
        # Keep only top k
        volume_list = volume_list[:k]
    df_topk = pd.DataFrame(volume_list, columns=[TICKER, VOLUME])
    return df_topk

def count_stock_by_month(ticker_list, start_date, end_date):
    '''
    Count number of tickers by month
    '''
    df_history = join_history(ticker_list, start_date, end_date, 
                              fill_missing=False)
    df_count = df_history.count(axis=1)
    df_monthly = df_count.resample('ME').max()
    return df_monthly

def save_temp_file(df_data, filename):
    '''
    Save DataFrame to a temporary file
    '''
    temp_dir = tempfile.gettempdir()
    filepath = temp_dir + '/' + filename + '.csv'
    df_data.to_csv(filepath)
    return filepath

def load_temp_file(filename):
    '''
    Load DataFrame from a temporary file
    '''
    temp_dir = tempfile.gettempdir()
    filepath = temp_dir + '/' + filename + '.csv'
    try:
        df_data = pd.read_csv(filepath, parse_dates=True,
                              index_col=DATE)
        return df_data
    except:
        return None

def test():
    '''
    Test data functions
    '''
    logger.setLevel(logging.DEBUG)
    df = get_ifix_series()
    logger.debug('IFIX: %s rows', len(df))
    df = _get_selic_series()
    logger.debug('SELIC: %s rows', len(df))
    fii_list = get_reit_list(only_valid=True)
    df = topk_by_volume(fii_list, 10, '2024-01-01', '2024-12-31')
    logger.debug('Top 10 by volume:')
    topk_list = df[TICKER].tolist()
    logger.debug('%s', topk_list)
    logger.debug('Joining data for top 10 REITs (2024).')
    df = join_history(topk_list,
                   '2024-01-01', '2024-12-31')
    logger.debug('\n%s', df)
    logger.debug('Counting REITs by month (2010-2024).')
    df = count_stock_by_month(fii_list,
                        '2010-01-01', '2024-12-31')
    logger.debug('Count by month:')
    df.to_csv('data/fiis_count.csv')
    logger.debug('\n%s', df)

def _get_arguments():
    '''
    Get arguments
    '''
    parser = argparse.ArgumentParser('OPBrReit Data Module')
    parser.add_argument('-d', '--download', action="store_true",
                        default=False,
                        help='Download data')
    parser.add_argument('-t', '--test', action="store_true",
                        default=False,
                        help='Test data functions')
    args = parser.parse_args()
    return args

def main():
    '''
    Main function
    '''
    args = _get_arguments()
    if args.download:
        download_all()
        update_valid_reit_list()
    if args.test:
        test()

if __name__ == '__main__':
    main()