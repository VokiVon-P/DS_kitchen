from utils.explogging import get_exp_logging, set_exp_logging_params

set_exp_logging_params('logs/ASPP_segmentation', True)
exp_lg = get_exp_logging()
exp_lg.start_experiment()

exp_lg.end_experiment()
