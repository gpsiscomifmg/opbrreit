# -*- coding: utf-8 -*-
'''
Universal logger module
'''

import logging
from pathlib import Path
from os import sep
from constants import *

# Log file
LOG_FILE = str(Path.home()) + sep + APP_NAME + '.log'
# Log level
LOG_LEVEL = logging.INFO

logger = logging.getLogger(APP_NAME)
if not logger.handlers:
    # Output string format
    str_format = '%(asctime)s - %(module)s - %(levelname)s - %(message)s'
    log_format = logging.Formatter(str_format)
    # Log file
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_format)
    # Console log
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
