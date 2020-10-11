import joblib
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from helper.config import *
from helper.help_data import load_data
from helper.help_time import time_it

RANDOM_STATE = 42


@time_it
def run_cv(estimator, cv, X, y, scoring=['f1', 'recall', 'precision'], model_name=""):
    cv_res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    logger.info("%s: %s = %0.2f (+/- %0.2f)" % (model_name,
                                          scoring[0],
                                          cv_res['test_f1'].mean(),
                                          cv_res['test_f1'].std()))
    logger.info("%s: %s = %0.2f (+/- %0.2f)" % (model_name,
                                          scoring[1],
                                          cv_res['test_precision'].mean(),
                                          cv_res['test_precision'].std()))
    logger.info("%s: %s = %0.2f (+/- %0.2f)" % (model_name,
                                          scoring[2],
                                          cv_res['test_recall'].mean(),
                                          cv_res['test_recall'].std()))


@time_it
def run_grid_search(estimator, X, y, params_grid, cv, scoring='f1'):
    gsc = GridSearchCV(estimator, params_grid, scoring=scoring, cv=cv, n_jobs=-1)

    gsc.fit(X, y)
    logger.info("Best %s score: %.2f" % (scoring, gsc.best_score_))
    logger.info()
    logger.info("Best parameters set found on development set:")
    logger.info()
    logger.info(gsc.best_params_)
    logger.info()
    logger.info("Grid scores on development set:")
    logger.info()

    for i, params in enumerate(gsc.cv_results_['params']):
        logger.info("%0.3f (+/-%0.03f) for %r"
              % (gsc.cv_results_['mean_test_score'][i], gsc.cv_results_['std_test_score'][i], params))

    logger.info()

    return gsc


@time_it
def load_data_for_train():
    path = PATH_DATASET + 'dataset_train.csv'
    df_train = load_data(path, sep=';')
    assert isinstance(df_train, pd.DataFrame)

    X = df_train.drop(['user_id', 'is_churned'], axis=1)
    y = df_train['is_churned']
    return X, y


@time_it
def save_model(model: object, path_to_save=FILE_MODEL):
    logger.info(path_to_save)
    try:
        joblib.dump(model, path_to_save)
    except Exception as ex:
        raise Exception('Ошибка при записи модели:' + path_to_save, ex)

    logger.info('Модель успешно сохранена!')


@time_it
def scale_balance_train(X_tr, y_tr):
    m_scaler = MinMaxScaler()
    # m_scaler = StandardScaler()
    X_sc = m_scaler.fit_transform(X_tr)
    save_model(m_scaler, FILE_SCALER)
    X_train_balanced, y_train_balanced = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.2).fit_sample(X_sc, y_tr)
    return X_train_balanced, y_train_balanced


@time_it
def train_model(X_train, y_train):
    kfold_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    xgb_estimator = xgb.XGBClassifier(max_depth=3,
                                      n_estimators=100,
                                      learning_rate=0.1,
                                      nthread=5,
                                      subsample=1.,
                                      colsample_bytree=0.5,
                                      min_child_weight=3,
                                      reg_alpha=0.,
                                      reg_lambda=0.,
                                      seed=RANDOM_STATE,
                                      missing=1e10,
                                      tree_method='gpu_hist',
                                      n_jobs=-1)
    run_cv(xgb_estimator, kfold_cv, X_train, y_train, model_name="Base")

    xgb_pipe = make_pipeline(
        SelectFromModel(LogisticRegression(solver='liblinear', penalty='l1', random_state=RANDOM_STATE),
                        threshold=1e-5),
        xgb.XGBClassifier(n_estimators=500,
                          max_depth=7,
                          learning_rate=0.1,
                          nthread=5,
                          colsample_bytree=0.5,
                          subsample=0.75,
                          reg_alpha=0.,
                          reg_lambda=0.,
                          seed=RANDOM_STATE,
                          missing=1e10,
                          tree_method='gpu_hist',
                          verbosity=2,
                          n_jobs=-1)
    )

    # TODO включить после тестирования функционала

    # # print([k for k in xgb_pipe.get_params().keys()])

    # param_grid = {
    #         # 'xgbclassifier__colsample_bytree': [0.5, 0.8],
    #         'xgbclassifier__subsample': [0.7, 0.5],
    #         'xgbclassifier__learning_rate': [0.1, 0.01],
    #         'xgbclassifier__max_depth': [5, 7, 9],
    #         'xgbclassifier__n_estimators': [400, 500, 700]
    # }
    #
    # xgb_gsc = run_grid_search(xgb_pipe, X_train, y_train, param_grid, kfold_cv)
    #
    # print(25 * '==')
    # print(xgb_gsc.best_params_)
    # print(25 * '==')
    #
    # xgb_final = xgb_gsc.best_estimator_

    xgb_final = xgb_pipe
    run_cv(xgb_final, kfold_cv, X_train, y_train, model_name="Final")

    # xgb_estimator.fit(X_train, y_train, eval_metric='aucpr', verbose=True)
    # estimator = xgb_estimator

    estimator = xgb_final
    estimator.fit(X_train, y_train)
    return estimator


def train():
    X, y = load_data_for_train()
    X_train_balanced, y_train_balanced = scale_balance_train(X, y)
    model = train_model(X_train_balanced, y_train_balanced)
    save_model(model=model)


# if __name__ == '__main__':
#     main()
