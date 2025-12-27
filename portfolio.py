# -*- coding: utf-8 -*-
'''
Portfolio functions
'''

import logging
import pandas as pd
from datetime import datetime, timedelta
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from constants import *
from data import join_history, topk_by_volume, get_reit_list
from logger import logger

def efficient_frontier(df_historical, risk_method, return_method, 
                       compounding=False, risk_free_rate=0.0):
    '''
    Calculate the efficient frontier
    '''
    # TODO: Maybe consider "compounding" parameter for geometric mean
    # Estimate future returns
    if return_method == CAPM_RETURN:
        df_return = expected_returns.return_model(
            df_historical,
            method=return_method,
            frequency=len(df_historical),
            compounding=compounding,
            risk_free_rate=risk_free_rate
        )
    else:
        df_return = expected_returns.return_model(
            df_historical,
            method=return_method,
            frequency=len(df_historical),
            compounding=compounding
        )
    # Estimate risk
    risk_matrix = risk_models.risk_matrix(
        df_historical,
        method=risk_method,
        frequency=len(df_historical)
    )
    # Fix non-positive definite risk matrix
    # "Spectral" method: set negative eigenvalues to zero and rebuilds matrix
    risk_matrix = risk_models.fix_nonpositive_semidefinite(
        risk_matrix)
    # Reference to ECOS solver: https://web.stanford.edu/~boyd/papers/pdf/ecos_ecc.pdf
    ef = EfficientFrontier(df_return, risk_matrix, solver='ECOS')
    # Convex optimization problem after making a certain variable substitution
    # <http://web.math.ku.dk/~rolf/CT_FinOpt.pdf>
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

def test():
    '''
    Test portfolio functions
    '''
    start_hist = '2024-01-01'
    end_hist = '2024-12-31'
    start_eval = '2025-01-01'
    end_eval = '2025-03-31'
    full_list = get_reit_list(only_valid=True)
    df_topk = topk_by_volume(
        full_list, k=100,
        start_date=start_hist, end_date=end_hist)
    topk_list = df_topk[TICKER].tolist()
    logger.debug('Top tickers by volume: %s', df_topk)
    df_history = join_history(topk_list, start_hist, end_hist)
    logger.debug('Historical data:\n%s', df_history)
    logger.setLevel(logging.DEBUG)
    list_results = []
    for compounding in [False, True]:
        for risk_method in LIST_RISK_METHODS:
            for return_method in LIST_RETURN_METHODS:
                try:
                    logger.debug('Testing risk method: %s, return method: %s, compounding: %s',
                                risk_method, return_method, compounding)
                    portfolio, exp_return, volatility, sharpe_ratio = efficient_frontier(
                        df_history,
                        risk_method=risk_method,
                        return_method=return_method,
                        compounding=compounding,
                        risk_free_rate=0.02,
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
                        compounding,
                        exp_return,
                        volatility,
                        sharpe_ratio,
                        portfolio_return[PORTFOLIO].values[0]
                    )
                    list_results.append(result_tuple)
                except Exception as e:
                    logger.error('Error for risk method: %s, return method: %s - %s',
                                risk_method, return_method, e)

    df_results =  pd.DataFrame(list_results, columns=[
        'Risk Method', 'Return Method', 'Compounding', 'Expected Return', 'Volatility', 'Sharpe Ratio', 'Portfolio Return'])
    logger.debug('Results:\n%s', df_results)
if __name__ == '__main__':
    test()
