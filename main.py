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


        
if __name__ == '__main__':

    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='处理配置文件')
    parser.add_argument('--config_path', type=str, required=True, help='配置文件路径', default="../args/config.yaml")
    args = parser.parse_args()

    config = Config(args.config_path)
    config.print_attribute()

    logger_config = LoggerConfig(config.log_path)
    log = get_logger()
    log.info("=" * 50)

    log.info(f"随机数种子是{seed}")
    log.info(f"选择的模型是{config.model_name}")
    


    end_time = datetime.now()
    log.info(f"start_time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"end_time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    log.info(f"总耗时为：{strftime('%H:%M:%S', gmtime(time_diff.total_seconds()))}")