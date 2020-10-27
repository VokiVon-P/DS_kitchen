import time
from datetime import datetime
import pandas as pd

from helper.config import *
from helper.help_time import time_format


def etl_loop(func):
    """
    Декоратор для запуска функций преобразования
    :param func:
    :return:
    """

    def _func(*args, **kwargs):
        _max_iter_cnt = 100
        for i in range(_max_iter_cnt):
            try:
                start_t = time.time()
                res = func(*args, **kwargs)
                run_time = time_format(time.time() - start_t)
                logger.info('Run time "{}": {}'.format(func.__name__, run_time))
                return res
            except Exception as er:
                run_time = time_format(time.time() - start_t)
                logger.info('Run time "{}": {}'.format(func.__name__, run_time))
                logger.info('-' * 50)
                logger.info(er, '''Try № {}'''.format(i + 1))
                logger.info('-' * 50)
        raise Exception('Max error limit exceeded: {}'.format(_max_iter_cnt))

    return _func


@etl_loop
def build_dataset_raw(churned_start_date,
                      churned_end_date,
                      inter_list,
                      raw_data_path,
                      dataset_raw_filename,
                      mode='train'):
    start_t = time.time()
    logger.info('Start reading csv files: {}'.format(datetime.now()))
    sample = pd.read_csv('{}sample.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    profiles = pd.read_csv('{}profiles.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    payments = pd.read_csv('{}payments.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    reports = pd.read_csv('{}reports.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    abusers = pd.read_csv('{}abusers.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    logins = pd.read_csv('{}logins.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    pings = pd.read_csv('{}pings.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    sessions = pd.read_csv('{}sessions.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')
    shop = pd.read_csv('{}shop.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'], encoding='utf-8')

    logger.info('Run time (reading csv files): {}'.format(time_format(time.time() - start_t)))
    # -----------------------------------------------------------------------------------------------------
    logger.info('NO dealing with outliers, missing values and categorical features...')
    # -----------------------------------------------------------------------------------------------------
    # На основании дня отвала (last_login_dt) строим признаки, которые описывают активность игрока перед уходом

    logger.info('Creating dataset...')
    # Создадим пустой датасет - в зависимости от режима построения датасета - train или test
    if mode == 'train':
        dataset = sample.copy()[['user_id', 'is_churned', 'level', 'donate_total']]
    elif mode == 'test':
        dataset = sample.copy()[['user_id', 'level', 'donate_total']]

    # Пройдемся по всем источникам, содержащим "динамичекие" данные
    for df in [payments, reports, abusers, logins, pings, sessions, shop]:

        # Получим 'day_num_before_churn' для каждого из значений в источнике для определения недели
        data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
        data['day_num_before_churn'] = 1 + (data['login_last_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) -
                                            data['log_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))).apply(
            lambda x: x.days)
        df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

        # Для каждого признака создадим признаки для каждого из времененно интервала (в нашем примере 4 интервала по 7 дней)
        features = list(set(data.columns) - {'user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn'})
        logger.info(f'Processing with features: {features}')
        for feature in features:
            for i, inter in enumerate(inter_list):
                inter_df = data.loc[data['day_num_before_churn'].between(inter[0], inter[1], inclusive=True)]. \
                    groupby('user_id')[feature].mean().reset_index(). \
                    rename(index=str, columns={feature: feature + '_{}'.format(i + 1)})
                df_features = pd.merge(df_features, inter_df, how='left', on='user_id')

        # Добавляем построенные признаки в датасет
        dataset = pd.merge(dataset, df_features, how='left', on='user_id')

        logger.info('Run time (calculating features): {}'.format(time_format(time.time() - start_t)))

    # Добавляем "статические" признаки
    dataset = pd.merge(dataset, profiles, on='user_id')
    # ---------------------------------------------------------------------------------------------------------------------------
    dataset.to_csv(dataset_raw_filename, sep=FILE_DATASET_SEP, index=False)
    logger.info('Dataset is successfully built and saved to {}, run time "build_dataset_raw": {}'. \
          format(dataset_raw_filename, time_format(time.time() - start_t)))


def load():

    build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                      churned_end_date=CHURNED_END_DATE,
                      inter_list=INTER_LIST,
                      raw_data_path=PATH_RAW_DATA_TRAIN,
                      dataset_raw_filename=FILE_DATASET_RAW_TRAIN,
                      mode='train')

    build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                      churned_end_date=CHURNED_END_DATE,
                      inter_list=INTER_LIST,
                      raw_data_path=PATH_RAW_DATA_TEST,
                      dataset_raw_filename=FILE_DATASET_RAW_TEST,
                      mode='test')


# if __name__ == '__main__':
#     main()
