"""
Обработка датасета на выбросы, пропуски и категориальные признаки
"""

import time

from helper.config import *
from helper.help_data import load_data
from helper.help_time import time_format
from helper.logging import logger


def prepare_dataset(dataset,
                    inter_list,
                    dataset_filename
                    ):
    start_t = time.time()
    logger.debug('Dealing with missing values, outliers, categorical features...')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1

    # Пинги
    for period in range(1, len(inter_list) + 1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()

    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv(dataset_filename, sep=FILE_DATASET_SEP, index=False)

    logger.debug('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'.
          format(dataset_filename, time_format(time.time() - start_t)))


def prepare():
    train = load_data(FILE_DATASET_RAW_TRAIN, sep=FILE_DATASET_SEP)
    test = load_data(FILE_DATASET_RAW_TEST, sep=FILE_DATASET_SEP)

    logger.debug(f'Размерность загруженных файлов: {train.shape}, {test.shape}')

    prepare_dataset(dataset=train, inter_list=INTER_LIST, dataset_filename=FILE_DATASET_MODEL_TRAIN)
    prepare_dataset(dataset=test, inter_list=INTER_LIST, dataset_filename=FILE_DATASET_MODEL_TEST)


# if __name__ == '__main__':
#     main()
