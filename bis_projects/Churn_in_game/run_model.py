from argparse import ArgumentParser

from ETL.etl_load import load
from ETL.etl_prepare import prepare
from helper.logging import logger
from models.predict import predict
from models.train import train

parser = ArgumentParser()
parser.add_argument(
    '-m', '--mode', type=str, required=False,
    help='Sets mode for model. [load, prepare, train, predict, all]'
)

params = {}
args = parser.parse_args()
if args.mode:
    params['mode'] = args.mode


def run_model():
    mode = params.get('mode')

    if not mode:
        logger.warning('Не указан режим работы приложения! ключ --mode [load, prepare, train, predict, all] ')
        logger.info('Пример: python run_model.py --mode train ')
        logger.info('По умолчанию режим:  predict  ')
        mode = 'predict'

    logger.info(f'Приложение запущено в режиме: { mode } ')

    if mode == 'predict':
        predict()
    elif mode == 'prepare':
        prepare()
    elif mode == 'load':
        load()
    elif mode == 'train':
        train()
    elif mode == 'all':
        load()
        prepare()
        train()
        predict()
    else:
        logger.warning(f'Режим MODE = {mode} не распознан!')

    logger.info(f'Приложение успешно завершено в режиме: {mode} ')


if __name__ == '__main__':
    run_model()
