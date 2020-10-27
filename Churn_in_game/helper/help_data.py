import pandas as pd

from helper.logging import logger


def load_data(path_to_load: str, **kwargs) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_load, **kwargs)
    except Exception as err:
        err_text = 'Ошибка при загрузке файла:' + path_to_load
        logger.exception(err_text)
        raise Exception(err_text, err)

    logger.info(f'Файл {path_to_load} загружен')
    return df
