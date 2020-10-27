import os
import functools
from datetime import datetime
from time import time
import json


def _get_experiment_log_dir(base_log_dir: str = "logs",
                            is_time_based: bool = 'True') -> str:
    """
    Создает и возвращает директорию для логирования

    :param base_log_dir: Директория для логов
    :param is_time_based: True - создаются поддиректории основанные на текущем времени
    :return: директория к логам
    """
    if is_time_based:
        start_time_str = datetime.today().strftime('%d_%m_%Y__%H_%M_%S')
        log_dir = f"{base_log_dir}/log_{start_time_str}"
    else:
        log_dir = base_log_dir

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


class ExpLogging:
    """
    Класс-singleton для логирования экспериментов
    """
    __instance = None
    __is_init = False
    # TODO: Перенести дальнейшие настройки в экземпляр - init()
    __base_log_dir: str = "logs"
    __is_base_dir_timed: bool = True
    __params_filename: str = 'params.json'
    __runtime_filename: str = 'runtime.json'

    def __new__(cls, *args, **kwargs):
        """ Стандартная реализация singleton """
        if not cls.__instance:
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        """ Инициализируем только первый инстанс"""
        if not self.__class__.__is_init:
            self.log_dir = _get_experiment_log_dir(self.__base_log_dir,
                                                   self.__is_base_dir_timed)
            self.log_params = {}
            self.global_start_time = None
            self.runtime_dict = {}
            self.__class__.__is_init = True

    def __call__(self, func: callable, *args, **kwargs):
        """
        Вызываем func и логируем его параметры и время выполнения
        :param func: callable
        :param args:
        :param kwargs:
        :return: результат выполнения func
        """
        return self.log_task_func(func)(*args, **kwargs)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_params()
        self.save_runtime_log()

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(log_dir: {self.log_dir!r}'
                f', log_params: {self.log_params})')

    @classmethod
    def set_logging_params(cls, log_dir: str, is_time_based: bool):
        """
        Устанавливает базовую директорию для логгера.\n
        !Только если экземпляр не был создан!
        """
        if not cls.__instance:
            cls.__base_log_dir = log_dir
            cls.__is_base_dir_timed = is_time_based

    def update_params(self, params: dict):
        """
        Добавление или изменение параметров эксперимента для логгирования

        :param params: словарь параметров
        :return: None
        """
        if params:
            self.log_params.update(params)

    def save_params(self, json_name: str = 'params.json'):
        sorted_params_dict = {k: self.log_params[k] for k
                              in sorted(self.log_params.keys())}
        params_filepath = os.path.join(self.log_dir, json_name)
        json.dump(sorted_params_dict, open(params_filepath, 'w'), indent=4)

    def start_experiment(self):
        self.global_start_time = time()
        self.runtime_dict.update({'START_EXPERIMENT': datetime.today().strftime('%d_%m_%Y__%H_%M_%S')})

    def end_experiment(self):
        self.log_task_end('global_experiment_runtime', self.global_start_time)
        self.runtime_dict.update({'END_EXPERIMENT': datetime.today().strftime('%d_%m_%Y__%H_%M_%S')})
        self.save_params()
        self.save_runtime_log()

    @staticmethod
    def log_task_start():
        return time()

    def log_task_end(self, task_name: str, task_start_time: float):
        task_runtime = time() - task_start_time
        self.runtime_dict.update({task_name: task_runtime})

    def save_runtime_log(self, runtime_file: str = 'runtime.json'):
        runtime_filepath = os.path.join(self.log_dir, runtime_file)
        json.dump(self.runtime_dict,
                  open(runtime_filepath, 'w'), indent=4)

    def log_task_func(self, func: callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = self.log_task_start()
            self.update_params({func.__name__: func.__dict__})
            func(*args, **kwargs)
            self.log_task_end(func.__name__, start_time)
            self.save_runtime_log()
        return wrapper


def set_exp_logging_params(logs_base_dir: str, is_time_based: bool):
    """
    Установить базувую директорию для логов - если экземпляр еще не создан!

    :param logs_base_dir: возможно установить базовую директорию если экземпляр еще не был создан
    :param is_time_based: создавать ли поддиректории основанные да дате-времени запуска
    :return: None
    """
    if len(logs_base_dir):
        ExpLogging.set_logging_params(logs_base_dir, is_time_based)


def get_exp_logging() -> ExpLogging:
    """
    Получить логгер

    :return: глобальный экземпляр логгера
    """
    return ExpLogging()


# хороший снипет - отложить
# sorted_params_dict = {k: params.__dict__[k] for k
#                       in sorted(params.__dict__.keys())}


if __name__ == '__main__':
    pass
