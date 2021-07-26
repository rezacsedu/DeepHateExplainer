import os
import json
import logging.config
import logging
import requests
import multiprocessing_logging

# LOGGER's
LOGGER_DEBUG_FILE = '../DeepHateLingo/training-script/xlm-roberta/logs/debug.log'
LOGGER_INFO_FILE = '../DeepHateLingo/training-script/xlm-roberta/logs/info.log'
LOGGER_ERROR_FILE = '../DeepHateLingo/training-script/xlm-roberta/logs/error.log'
lb_logging_mode = "development"


def get_abspath(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

def load_config(config_file):
    """ Loads json config from a file """
    abspath = get_abspath(config_file)
    if os.path.exists(abspath):
        with open(abspath) as json_data_file:
            config = json.load(json_data_file)
            return config
    return

def get_module_logger(mod_name):
    """
    Load the logger module from logging config json
    """
    config = load_config('logging.json')
    if config:
        if LOGGER_DEBUG_FILE:
            config['handlers']['console']['filename'] = LOGGER_DEBUG_FILE
        if LOGGER_INFO_FILE:
            config['handlers']['info_file_handler']['filename'] = LOGGER_INFO_FILE
        if LOGGER_ERROR_FILE:
            config['handlers']['error_file_handler']['filename'] = LOGGER_ERROR_FILE
        
        if lb_logging_mode == "development":
            config['root']['level'] = "DEBUG"        
        else:
            config['root']['level'] = "INFO"
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level="INFO")

    multiprocessing_logging.install_mp_handler()
    logger = logging.getLogger(mod_name)
    return logger