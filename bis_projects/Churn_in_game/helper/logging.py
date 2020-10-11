import logging
import logging.config
from os.path import abspath
import yaml
# from pprint import pprint

FILE_LOGGING_CFG = abspath('./config/logging_cfg.yaml')


def load_logging_config(filename=None):

    if not filename:
        filename = FILE_LOGGING_CFG

    with open(filename, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def save_logging_config(log_conf: dict, filename=None):

    if not filename:
        filename = FILE_LOGGING_CFG

    try:
        with open(filename, 'w') as f_n:
            yaml.dump(log_conf, f_n, allow_unicode=True, default_flow_style=False)

    except Exception as err:
        logger.exception(f"Ошибка записи файла {filename}\n")
        raise err


load_logging_config()
logger = logging.getLogger('model_logger')
logger.info('logger configured and started')
