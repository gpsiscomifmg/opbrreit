# -*- coding: utf-8 -*-
'''
Experiments
'''

import itertools
import logging
import pandas as pd
import time
from datetime import datetime
from constants import *
from data import join_history, get_selic, get_reit_list
from portfolio import historical_interval, efficient_frontier
from logger import logger

# TODO: Maybe consider precision to deal with rounding errors from Yahoo Finance data

EXP_COL_ID = [YEAR, MONTH, RISK_METHOD, RETURN_METHOD]
EXP_COL_LIST = [
    YEAR, MONTH, RISK_METHOD, RETURN_METHOD, EXPECTED_RETURN, VOLATILITY,
    SHARPE_RATIO, RISK_FREE_RATE, INPUT_SIZE, PORTFOLIO_SIZE, EVALUATION_TIME,
    PORTFOLIO
]

def gen_period_list(start, end, step=1):
    '''
    Generate date list
    '''
    year, month = start
    period_list = []
    while year < end[0] or (year == end[0] and month <= end[1]):
        period_list.append((year, month))
        month += step
        if month > 12:
            month -= 12
            year += 1
    return period_list

def gen_experiment_list(start, end, step=1):
    '''
    Generate experiment list
    '''
    period_list = gen_period_list(start, end, step)
    experiment_list = list(
        itertools.product(
            period_list,
            LIST_RISK_METHODS,
            LIST_RETURN_METHODS,
            [False, True]  # Compounding
        )
    )
    return experiment_list

def resume_experiments():
    '''
    Resume experiments
    '''
    logger.debug('Resuming experiments')
    try:
        logger.debug('Loading experiments from file: %s', FILE_EXP_INITIAL)
        df_experiments = pd.read_csv(FILE_EXP_INITIAL, sep='|')
        df_concluded = df_experiments[EXP_COL_ID].drop_duplicates(ignore_index=True)
        concluded_set = set(list(
            df_concluded.itertuples(index=False, name=None)
        ))
        logger.debug('Concluded experiments: %d', len(concluded_set))
        return concluded_set, df_experiments
    except:
        logger.debug('No previous experiments found.')
        df_experiments = pd.DataFrame(columns=EXP_COL_LIST)
        return set(), df_experiments

def run_experiment(df_full_history, experiment):
    '''
    Run a single experiment
    '''
    start_hist, end_hist = historical_interval(
        experiment[YEAR], experiment[MONTH])
    df_history = df_full_history.loc[start_hist:end_hist].copy()
    df_history.dropna(axis=1, how='all', inplace=True)
    df_history.ffill(inplace=True)
    df_history.bfill(inplace=True)
    num_assets = len(df_history.columns)
    selic = get_selic(
        experiment[YEAR], experiment[MONTH])
    start_time = time.time()
    portfolio, expected_return, volatility, sharpe_ratio \
        = efficient_frontier(
            df_history,
            risk_method=experiment[RISK_METHOD],
            return_method=experiment[RETURN_METHOD],
            risk_free_rate=selic
        )
    eval_time = time.time() - start_time
    result = {
        EXPECTED_RETURN: expected_return, VOLATILITY: volatility,
        SHARPE_RATIO: sharpe_ratio, RISK_FREE_RATE: selic,
        INPUT_SIZE: num_assets, PORTFOLIO_SIZE: len(portfolio),
        EVALUATION_TIME: eval_time, PORTFOLIO: portfolio
        }
    return result

def run_all_experiments(df_full_history, start, end):
    '''
    Run experiments
    '''
    experiment_list = gen_experiment_list(start, end)
    concluded_set, df_experiments = resume_experiments()
    for experiment in experiment_list:
        experiment = {
            YEAR: experiment[0][0],
            MONTH: experiment[0][1],
            RISK_METHOD: experiment[1],
            RETURN_METHOD: experiment[2]
        }
        experiment_tuple = tuple(experiment.values())
        if experiment_tuple in concluded_set:
            continue
        try:
            logger.debug('Running experiment: %s', experiment)
            result = run_experiment(df_full_history, experiment)
            result.update(experiment)
            logger.debug('Experiment result: %s', result)
        except Exception as e:
            logger.error('Error in experiment %s - %s', experiment, e)
            result = {
                EXPECTED_RETURN: None, VOLATILITY: None, SHARPE_RATIO: None, 
                RISK_FREE_RATE: None, INPUT_SIZE: None, PORTFOLIO_SIZE: None, 
                EVALUATION_TIME: None, PORTFOLIO: None
            }
            result.update(experiment)
        concluded_set.add(experiment_tuple)
        df_result = pd.DataFrame([result])
        df_experiments = pd.concat([df_experiments, df_result])
        df_experiments.to_csv(FILE_EXP_INITIAL, index=False,
                                sep='|')

def main():
    '''
    Main function
    '''
    logger.setLevel(logging.DEBUG)
    fii_list = get_reit_list(only_valid=True)
    start = (2011, 1)
    end = (2025, 6)
    start_history = datetime(start[0]-1, start[1], 1).strftime(DATE_FORMAT)
    end_history = datetime(end[0], end[1], 28).strftime(DATE_FORMAT)
    logger.debug('Loading full history from %s to %s', start_history, end_history)
    df_full_history = join_history(
        fii_list,
        start_date=start_history,
        end_date=end_history
    )
    run_all_experiments(
        df_full_history,
        start=start,
        end=end
    )

if __name__ == '__main__':
    main()