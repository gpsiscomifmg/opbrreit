# -*- coding: utf-8 -*-
'''
Constants
'''
import tempfile

APP_NAME = 'opbrreit'

DIR_DATA = 'data'
DIR_STOCK = 'stock'
# Files
FILE_REITS = 'data/b3reits.txt'
FILE_REITS_VALID = 'data/b3reits_valid.txt'
FILE_IFIX = 'data/ifix.csv'
FILE_SELIC = 'data/selic.csv'
FILE_EXP_INITIAL = 'data/exp_initial.csv'

# URLs
URL_IFIX = 'https://sistemaswebb3-listados.b3.com.br/indexStatisticsProxy/IndexCall/GetPortfolioDay/'

# Asset dataframe columns
TICKER = 'ticker'
DATE = 'Date'
OPEN = 'Open'
HIGH = 'High'
LOW = 'Low'
CLOSE = 'Close'
VOLUME = 'Volume'
DIVIDENDS = 'Dividends'
STOCK_SPLITS = 'Stock Splits'

# Indexes
IFIX = 'IFIX'
SELIC = 'SELIC'
IFIX_START_YEAR = 2010
SELIC_START_YEAR = 2000

# Experiment columns
YEAR = 'Year'
MONTH = 'Month'
RISK_METHOD = 'Risk Method'
RETURN_METHOD = 'Return Method'
EXPECTED_RETURN = 'Expected Return'
VOLATILITY = 'Volatility'
SHARPE_RATIO = 'Sharpe Ratio'
REAL_RETURN = 'Real Return'
RISK_FREE_RATE = 'Risk Free Rate'
EVALUATION_TIME = 'Evaluation Time'
INPUT_SIZE = 'Input Size'
PORTFOLIO_SIZE = 'Portfolio Size'
PORTFOLIO = 'Portfolio'

# Risk methods
SAMPLE_COV = 'sample_cov'
SEMICOVARIANCE = 'semicovariance'
EXP_COV = 'exp_cov'
LEDOIT_WOLF_CONSTANT_VARIANCE = 'ledoit_wolf_constant_variance'
LEDOIT_WOLF_SINGLE_FACTOR = 'ledoit_wolf_single_factor'
LEDOIT_WOLF_CONSTANT_CORRELATION = 'ledoit_wolf_constant_correlation'
ORACLE_APPROXIMATING = 'oracle_approximating'
LIST_RISK_METHODS = [SAMPLE_COV, SEMICOVARIANCE, EXP_COV,
                     LEDOIT_WOLF_CONSTANT_VARIANCE, LEDOIT_WOLF_SINGLE_FACTOR,
                     LEDOIT_WOLF_CONSTANT_CORRELATION, ORACLE_APPROXIMATING]
# Return methods
MEAN_HISTORICAL_RETURN = 'mean_historical_return'
EMA_HISTORICAL_RETURN = 'ema_historical_return'
CAPM_RETURN = 'capm_return'
LIST_RETURN_METHODS = [MEAN_HISTORICAL_RETURN, EMA_HISTORICAL_RETURN, 
                       CAPM_RETURN]

DATE_FORMAT = '%Y-%m-%d'

# Experiment parameters
# Times in months
TIME_HISTORY = 12
