import random
from time import gmtime
from time import strftime
from datetime import datetime
import joblib
from models import *
import torch
import argparse
from config import Config
from logger_config import LoggerConfig, get_logger
from exp.exp_main import Exp_Main
import numpy as np


        
if __name__ == '__main__':

    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, default=None, help='Path to yaml config')
    args, _ = parser.parse_known_args()
    config = Config(config_path=args.config_path, args=args)

    logger_config = LoggerConfig(config.log_path)
    config.print_attribute()
    log = get_logger()
    log.info("=" * 50)

    fix_seed = config.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False

    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    Exp = Exp_Main

    if config.is_training:
        for ii in range(config.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                config.model_id,
                config.model,
                config.data,
                config.features,
                config.seq_len,
                config.label_len,
                config.pred_len,
                config.d_model,
                config.n_heads,
                config.e_layers,
                config.d_layers,
                config.d_ff,
                config.factor,
                config.embed,
                config.distil,
                config.des, ii)

            exp = Exp(config)  # set experiments
            log.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            log.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if config.do_predict:
                log.info('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(config.model_id,
                                                                                                    config.model,
                                                                                                    config.data,
                                                                                                    config.features,
                                                                                                    config.seq_len,
                                                                                                    config.label_len,
                                                                                                    config.pred_len,
                                                                                                    config.d_model,
                                                                                                    config.n_heads,
                                                                                                    config.e_layers,
                                                                                                    config.d_layers,
                                                                                                    config.d_ff,
                                                                                                    config.factor,
                                                                                                    config.embed,
                                                                                                    config.distil,
                                                                                                    config.des, ii)

        exp = Exp(config)  # set experiments
        log.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


    end_time = datetime.now()
    log.info(f"start_time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"end_time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    log.info(f"总耗时为：{strftime('%H:%M:%S', gmtime(time_diff.total_seconds()))}")