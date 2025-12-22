# -*- coding: utf-8 -*-
'''
Data functions
'''

import requests
import logging
import yfinance as yf
import pandas as pd
from os import makedirs
from base64 import b64encode
from tenacity import retry, wait_random, stop_after_attempt
from tqdm import tqdm
from platformdirs import user_cache_dir
from bcb import sgs
from constants import *
from logger import logger

def _to_float(text):
    '''
    Convert text to float
    '''
    text = text.replace('.', '')
    text = text.replace(',', '.')
    return float(text)

def fetch_url(url):
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
    content = fetch_url(url)
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
                    IFIX: _to_float(value)
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

def get_ifix():
    '''
    Get IFIX data from file or try to download it
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
        df_current = sgs.get({SELIC: 11},
                           start=start, end=end)
        year += 10
        df_selic = pd.concat([df_selic, df_current])
    df_selic.to_csv(FILE_SELIC)

def get_selic():
    '''
    Get SELIC data from file or try to download it
    '''
    df_selic = None
    try:
        df_selic = pd.read_csv(FILE_SELIC, parse_dates=True,
                               index_col=DATE)
    except:
        _download_selic()
        df_selic = pd.read_csv(FILE_SELIC, parse_dates=True,
                               index_col=DATE)
    return df_selic

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

def save_valid_reit_list(reit_list):
    '''
    Save the valid list of REITs
    '''
    with open(FILE_REITS_VALID, 'w') as f:
        for reit in reit_list:
            f.write(reit + '\n')

def get_cache_dir():
    '''
    Get cache directory for the application
    '''
    cache_dir = user_cache_dir(APP_NAME)
    makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_filename(ticker, subdir=''):
    '''
    Get the filename for a ticker
    '''
    filedir = get_cache_dir()
    if subdir:
        filedir += '/' + subdir
        makedirs(filedir,
                 exist_ok=True)
    filename = filedir + '/' + ticker + '.csv'
    return filename

def empty_data():
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
def fetch_stock_data(ticker):
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
        df_data = empty_data()
    return df_data

def download_stock_list(ticker_list):
    '''
    Fetch data for a list of tickers
    '''
    success_list = []
    # Use tqdm for parallel fetching with progress bar
    for ticker in tqdm(ticker_list, desc='Fetching data'):
        try:
            df_data = fetch_stock_data(ticker)
            filename = get_filename(ticker, DIR_STOCK)
            df_data.to_csv(filename)
            success_list.append(ticker)
        except Exception as e:
            logger.error('Error fetching data for %s.', ticker)
            logger.error('%s', e)
    return success_list

def get_stock_data(ticker):
    '''
    Get stock data from cache
    '''
    filename = get_filename(ticker, DIR_STOCK)
    try:
        return pd.read_csv(filename, parse_dates=True, index_col=DATE)
    except:
        return empty_data()

def download_all():
    '''
    Test data functions
    '''
    logger.debug('Downloading all IFIX')
    _download_ifix()
    logger.debug('Downloading all SELIC')
    _download_selic()
    logger.debug('Downloading all REIT data')
    reit_list = get_reit_list()
    final_list = download_stock_list(reit_list)
    # print(f'Fetched data for {len(final_list)} REITs.')
    logger.debug('Saving valid REIT list. Total %s REITs.', len(final_list))
    save_valid_reit_list(final_list)

def join_data(ticker_list, start_date, end_date, field=CLOSE):
    '''
    Join data for multiple tickers
    '''
    df_joined = pd.DataFrame()
    for ticker in ticker_list:
        logger.debug('Joining data for %s.', ticker)
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
    df_joined.ffill(inplace=True)
    df_joined.bfill(inplace=True)
    return df_joined

def topk_by_volume(ticker_list, k=0, start_date=None, end_date=None):
    '''
    Get top k tickers by average volume
    '''
    volume_list = []
    for ticker in ticker_list:
        logger.debug('Calculating volume for %s.', ticker)
        df_data = get_stock_data(ticker)
        if df_data is None or df_data.empty:
            continue
        if start_date is not None and end_date is not None:
            df_data = df_data[start_date:end_date]
        # Consider VOLUME as VOLUME * CLOSE
        df_data[VOLUME] = df_data[VOLUME] * df_data[CLOSE]
        volume_sum = df_data[[VOLUME]].sum().values[0]
        volume_list.append((ticker, volume_sum))
        df_data = df_data[[VOLUME]]
    # Sort by volume descending
    volume_list.sort(key=lambda x: x[1], reverse=True)
    if k > 0:
        # Keep only top k
        volume_list = volume_list[:k]
    df_topk = pd.DataFrame(volume_list, columns=[TICKER, VOLUME])
    return df_topk

def test():
    '''
    Test data functions
    '''
    logger.setLevel(logging.DEBUG)
    # download_all()
    df = get_ifix()
    logger.debug('IFIX: %s rows', len(df))
    df = get_selic()
    logger.debug('SELIC: %s rows', len(df))
    fii_list = get_reit_list(only_valid=True)
    df = topk_by_volume(fii_list, k=10)
    logger.debug('Top 10 by volume:')
    topk_list = df[TICKER].tolist()
    logger.debug('%s', topk_list)
    logger.debug('Joining data for top 10 REITs.')
    df = join_data(topk_list,
                   '2020-01-01', '2024-12-31')
    logger.debug('Joined data:')
    logger.debug('\n%s', df)

if __name__ == '__main__':
    test()