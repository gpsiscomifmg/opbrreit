# -*- coding: utf-8 -*-
'''
Portfolio functions
'''

import logging
import pandas as pd
import warnings
import cvxpy as cp

from datetime import datetime, timedelta
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from constants import *
from data import join_history, topk_by_volume, get_reit_list, get_selic
from logger import logger

warnings.filterwarnings('ignore')

def efficient_frontier(df_historical, risk_method, return_method,
                       risk_free_rate=0.0):
    '''
    Calculate the efficient frontier
    '''
    # Estimate future returns
    if return_method == CAPM_RETURN:
        df_return = expected_returns.return_model(
            df_historical,
            method=return_method,
            frequency=len(df_historical),
            risk_free_rate=risk_free_rate
        )
    else:
        df_return = expected_returns.return_model(
            df_historical,
            method=return_method,
            frequency=len(df_historical),
        )
    # Estimate risk
    risk_matrix = risk_models.risk_matrix(
        df_historical,
        method=risk_method,
        frequency=len(df_historical)
    )
    # References to SCS solver: https://www.cvxgrp.org/scs/citing/
    ef = EfficientFrontier(df_return, risk_matrix, solver='SCS')

    # Convex optimization problem after making a certain variable substitution
    # Optimization Methods in Finance (2nd Ed.) by Gérard Cornuéjols, Javier Peña  and Reha Tütüncü (2018)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    # Get weights and build portfolio dictionary
    weights = ef.clean_weights()
    portfolio = {ticker: weight
                 for ticker, weight in weights.items() if weight > 0}
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return portfolio, expected_return, volatility, sharpe_ratio
    
def get_portfolio_returns(portfolio, start_date, end_date):
    '''
    Calculate portfolio returns over time
    '''
    # Get historical data for portfolio tickers
    df_history = join_history(portfolio.keys(), start_date, end_date)
    for ticker, weight in portfolio.items():
        df_history[ticker] = df_history[ticker] * weight
    df_history[PORTFOLIO] = df_history.sum(axis=1)
    df_history[PORTFOLIO] = df_history[PORTFOLIO].pct_change().fillna(0)
    return df_history[[PORTFOLIO]]

def historical_interval(year, month, duration=12):
    '''
    Calculate historical date interval
    '''
    end_date = datetime(year, month, 1) - timedelta(days=1)
    month -= duration
    if month <= 0:
        month += 12
        year -= 1
    start_date = datetime(year, month, 1)
    return (start_date.strftime(DATE_FORMAT), end_date.strftime(DATE_FORMAT))

def evaluation_interval(year, month, duration=3):
    '''
    Calculate evaluation date interval
    '''
    start_date = datetime(year, month, 1)
    month += duration
    if month > 12:
        month -= 12
        year += 1
    end_date = datetime(year, month, 1) - timedelta(days=1)
    return (start_date.strftime(DATE_FORMAT), end_date.strftime(DATE_FORMAT))


def gen_period_list(start, end, time_step=1):
    '''
    Generate period list
    '''
    year, month, _ = map(int, start.split('-'))
    end_year, end_month, _ = map(int, end.split('-'))
    period_list = []
    while (year < end_year) or (year == end_year and month <= end_month):
        month += time_step
        if month > 12:
            month -= 12
            year += 1
        period_list.append((year, month))
    return period_list

def test():
    '''
    Test portfolio functions
    '''
    year = 2012
    month = 8
    start_hist, end_hist = historical_interval(year, month, duration=12)
    start_eval, end_eval = evaluation_interval(year, month, duration=12)
    full_list = get_reit_list(only_valid=True)
    df_topk = topk_by_volume(
        full_list, k=100,
        start_date=start_hist, end_date=end_hist)
    topk_list = df_topk[TICKER].tolist()
    df_history = join_history(topk_list, start_hist, end_hist)
    logger.setLevel(logging.DEBUG)
    print(df_history.columns)
    logger.debug('Number of assets: %d', len(df_history.columns))
    list_results = []
    selic = get_selic(year, month)
    logger.debug('SELIC rate: %.4f', selic)
    for risk_method in LIST_RISK_METHODS:
        for return_method in LIST_RETURN_METHODS:
            try:
                logger.debug('Testing risk method: %s, return method: %s',
                             risk_method, return_method)
                portfolio, exp_return, volatility, sharpe_ratio = efficient_frontier(
                    df_history,
                    risk_method=risk_method,
                    return_method=return_method,
                    risk_free_rate=selic,
                )
                logger.debug('Portfolio: %s', portfolio)
                df_portfolio_returns = get_portfolio_returns(
                    portfolio,
                    start_eval,
                    end_eval
                )
                portfolio_return = df_portfolio_returns[-1:]
                result_tuple = (
                    risk_method,
                    return_method,
                    exp_return,
                    volatility,
                    sharpe_ratio,
                    portfolio_return[PORTFOLIO].values[0],
                    len(portfolio)
                )
                list_results.append(result_tuple)
            except Exception as e:
                logger.error('Error for %s, %s \n %s',
                             risk_method, return_method, e)
    df_results =  pd.DataFrame(list_results, columns=[
        RISK_METHOD, RETURN_METHOD, EXPECTED_RETURN, VOLATILITY, SHARPE_RATIO, 
        REAL_RETURN, PORTFOLIO_SIZE])
    df_results = df_results.round(4)
    print(f'Historical period: {start_hist} to {end_hist}')
    print(f'Evaluation period: {start_eval} to {end_eval}')
    print(df_results)

if __name__ == '__main__':
    test()
