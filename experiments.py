# -*- coding: utf-8 -*-
'''
Experiments
'''

import itertools
import pandas as pd
from datetime import datetime, timedelta
from constants import *
from data import join_history, topk_by_volume, load_temp_file, save_temp_file, get_selic, get_reit_list
from portfolio import historical_interval, evaluation_interval, efficient_frontier, get_portfolio_returns
from logger import logger
from portfolio import evaluation_interval

TIME_HISTORY = 12
TIME_EVALUATION = 3
TOPK = 100

# TODO: Maybe consider precision to deal with rounding errors from Yahoo Finance data


def gen_period_list(start, end, interval):
    '''
    Generate date list
    '''
    year, month, _ = map(int, start.split('-'))
    period_list = []
    while True:
        _, final_eval = evaluation_interval(year, month, 
                                            duration=TIME_EVALUATION)
        if final_eval > end:
            break
        period_list.append((year, month))
        month += interval
        if month > 12:
            month -= 12
            year += 1
    return period_list

def gen_experiment_list(start, end):
    '''
    Generate experiment list
    '''
    period_list = gen_period_list(start, end, TIME_EVALUATION)
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
    try:
        df_experiments = pd.read_csv(FILE_EXPERIMENTS, sep='|')
        df_concluded = df_experiments[
            [YEAR, MONTH, RISK_METHOD,RETURN_METHOD, COMPOUNDING]
            ].drop_duplicates(ignore_index=True)
        concluded_set = set(list(
            df_concluded.itertuples(index=False, name=None)
        ))
        return concluded_set, df_experiments
    except:
        df_experiments = pd.DataFrame(columns=[
            YEAR, MONTH, RISK_METHOD, RETURN_METHOD, COMPOUNDING,
            EXPECTED_RETURN, VOLATILITY, SHARPE_RATIO,
            REAL_RETURN, RISK_FREE_RATE
        ])
        return set(), df_experiments

def get_topk_fiis(fii_list, year, month, k):
    '''
    Get top k FIIs by volume in the given period
    '''
    topk_list = []
    df_topk = load_temp_file('df_topk_{}_{}_{}.csv'.format(
            year, month, k))
    if df_topk is None:
        start, end = historical_interval(
            year, month, duration=TIME_HISTORY)
        df_topk = topk_by_volume(
            fii_list, k=k, start_date=start, end_date=end)
        save_temp_file(df_topk, 'df_topk_{}_{}_{}.csv'.format(
            year, month, k))
    topk_list = df_topk[TICKER].tolist()
    return topk_list

def get_historical_data(fii_list, year, month):
    '''
    Get historical data for the given period
    '''
    df_history = load_temp_file('df_history_{}_{}.csv'.format(
            year, month))
    if df_history is None:
        start, end = historical_interval(
            year, month, duration=TIME_HISTORY)
        df_history = join_history(fii_list, start, end)
        save_temp_file(df_history, 'df_history_{}_{}.csv'.format(
            year, month))
    return df_history

def run_experiment(fii_list, experiment):
    '''
    Run a single experiment
    '''
    topk_list = get_topk_fiis(
        fii_list, experiment[YEAR], experiment[MONTH], k=TOPK)
    df_history = get_historical_data(
        topk_list, experiment[YEAR], experiment[MONTH])
    selic = get_selic(
        experiment[YEAR], experiment[MONTH])
    portfolio, expected_return, volatility, sharpe_ratio \
        = efficient_frontier(
            df_history,
            risk_method=experiment[RISK_METHOD],
            return_method=experiment[RETURN_METHOD],
            compounding=experiment[COMPOUNDING],
            risk_free_rate=selic
        )
    start_eval, end_eval = evaluation_interval(
        experiment[YEAR], experiment[MONTH], duration=TIME_EVALUATION)
    df_portfolio_returns = get_portfolio_returns(
        portfolio,
        start_eval,
        end_eval
    )
    portfolio_return = df_portfolio_returns[-1:]
    real_return = portfolio_return[PORTFOLIO].values[0]
    result = {
        EXPECTED_RETURN: expected_return,
        VOLATILITY: volatility,
        SHARPE_RATIO: sharpe_ratio,
        REAL_RETURN: real_return,
        RISK_FREE_RATE: selic,
        PORTFOLIO: portfolio,
        EXPECTED_RETURN: expected_return,
        VOLATILITY: volatility,
        SHARPE_RATIO: sharpe_ratio,
        REAL_RETURN: real_return
    }
    return result

def run_all_experiments(fii_list, start, end):
    '''
    Run experiments
    '''
    experiment_list = gen_experiment_list(start, end)
    concluded_set, df_experiments = resume_experiments()
    for experiment in experiment_list:
        if experiment in concluded_set:
            continue
        experiment = {
            YEAR: experiment[0][0],
            MONTH: experiment[0][1],
            RISK_METHOD: experiment[1],
            RETURN_METHOD: experiment[2],
            COMPOUNDING: experiment[3]
        }
        try:
            logger.debug('Running experiment: %s', experiment)
            result = run_experiment(fii_list, experiment)
            result.update(experiment)
            logger.debug('Experiment result: %s', result)
            experiment_tuple = (
                experiment[YEAR],
                experiment[MONTH],
                experiment[RISK_METHOD],
                experiment[RETURN_METHOD],
                experiment[COMPOUNDING]
            )
            concluded_set.add(experiment_tuple)
            df_result = pd.DataFrame([result])
            df_experiments = pd.concat([df_experiments, df_result])
            df_experiments.to_csv(FILE_EXPERIMENTS, index=False,
                                  sep='|')
        except Exception as e:
            logger.error('Error in experiment %s - %s', experiment, e)

def main():
    '''
    Main function
    '''
    fii_list = get_reit_list(only_valid=True)
    run_all_experiments(
        fii_list,
        start='2020-01-01',
        end='2025-12-31'
    )

if __name__ == '__main__':
    main()