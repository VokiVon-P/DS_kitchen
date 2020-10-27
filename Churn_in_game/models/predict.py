import os
import joblib
import pandas as pd

from helper.config import *
from helper.help_data import load_data
from helper.logging import logger


def load_data_for_predict():
    path = FILE_DATASET_MODEL_TEST

    df_predict = load_data(path, sep=FILE_DATASET_SEP)
    assert isinstance(df_predict, pd.DataFrame)

    X = df_predict.drop(['user_id'], axis=1)
    usersID = df_predict['user_id']
    return X, usersID


def load_model(path_to_load=FILE_MODEL):
    try:
        model = joblib.load(path_to_load)
    except Exception as err:
        err_text = 'Ошибка при загрузке модели:' + path_to_load
        logger.exception(err_text)
        raise Exception(err_text, err)

    logger.info(f'Модель {path_to_load} загружена')
    return model


def scale_predict(X_pr):
    m_scaler = load_model(FILE_SCALER)
    X_predict = m_scaler.transform(X_pr)
    return X_predict


def predict_model(X_predict):
    clf = load_model(FILE_MODEL)
    logger.debug(f'predict модель: {type(clf)}')
    y_pred = clf.predict_proba(X_predict)
    logger.debug(f'результат predict_proba = {y_pred.shape}')
    return y_pred


def make_result(y_predict, userID):
    # создаю результат классификатора на осное порога MODEL_THRESHOLD
    y_result = (~(y_predict[:, 1] < MODEL_THRESHOLD)).astype('int')
    d = {'user_id': userID, 'is_churned': y_result}
    df_res = pd.DataFrame(data=d)
    try:
        # если нет директории - создадим
        res_dir = os.path.dirname(FILE_MODEL_PREDICTION)
        os.makedirs(res_dir,exist_ok=True)

        df_res.to_csv(FILE_MODEL_PREDICTION, sep=FILE_DATASET_SEP, index=False)
    except Exception as err:
        logger.exception(f"Ошибка записи файла {FILE_MODEL_PREDICTION}\n")
        raise err

    logger.info(f'Результат работы модели {df_res.shape} записан в: {FILE_MODEL_PREDICTION}')



def predict():
    X_pr, userID = load_data_for_predict()
    X_pr_scaled = scale_predict(X_pr)
    y_pr = predict_model(X_pr_scaled)
    make_result(y_pr, userID)


# if __name__ == '__main__':
#     predict_model()