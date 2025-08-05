import logging
from datetime import datetime
import argparse
from logger_config import get_logger

import yaml


class Config(object):
    def __init__(self, config_file_path=None, args=None):
        """
        :param config_file_path: str, yaml配置文件路径
        :param args: argparse.Namespace或dict, 可允许命令行优先参数覆盖
        """

        self.args = args if args is not None else None

        # 加载yaml配置
        if config_file_path:
            with open(config_file_path, 'r', encoding='utf-8') as stream:
                self._configs = yaml.safe_load(stream)
        else:
            self._configs = {}

        '''step1: 基础参数'''
        self.date = datetime.now().strftime('%Y%m%d')
        self.version = datetime.now().strftime("%H%M%S")
        self.model_name = self.get_config('common', 'model_name')
        self.model_id = self.get_config('common', 'model_id')
        self.data = self.get_config('common', 'data')

        '''step2: 路径参数'''
        self.root_path = self.get_config('path', 'root_path')
        self.data_path = self.get_config('path', 'data_path')
        self.log_path = self.get_config('path', 'log_path')
        self.log_path = self.get_config('path', 'log_path').replace(".log", f"_{self.date}_{self.version}.log")

        '''step3: 模型相关参数'''
        mp = 'model_params'
        self.features = self.get_config(mp, 'features')
        self.random_seed = self.get_config(mp, 'random_seed')
        self.is_training = self.get_config(mp, 'is_training')
        self.seq_len = self.get_config(mp, 'seq_len')
        self.pred_len = self.get_config(mp, 'pred_len')
        self.enc_in = self.get_config(mp, 'enc_in')
        self.e_layers = self.get_config(mp, 'e_layers')
        self.n_heads = self.get_config(mp, 'n_heads')
        self.d_model = self.get_config(mp, 'd_model')
        self.d_ff = self.get_config(mp, 'd_ff')
        self.dropout = self.get_config(mp, 'dropout')
        self.fc_dropout = self.get_config(mp, 'fc_dropout')
        self.kernel_list = self.get_config(mp, 'kernel_list')
        self.period = self.get_config(mp, 'period')
        self.patch_len = self.get_config(mp, 'patch_len')
        self.stride = self.get_config(mp, 'stride')

        '''step4: 训练参数'''
        tp = 'train_params'
        self.des = self.get_config(tp, 'des')
        self.train_epochs = self.get_config(tp, 'train_epochs')
        self.patience = self.get_config(tp, 'patience')
        self.itr = self.get_config(tp, 'itr')
        self.batch_size = self.get_config(tp, 'batch_size')
        self.learning_rate = self.get_config(tp, 'learning_rate')

        # self.metric_path = self.get_config('path', 'metric_path').replace(".png", f"_{self.type}_{self.date}_{self.version}.png")
        # self.predict_show_plt_path = self.get_config('path', 'predict_show_plt_path').replace(".png",
        #                                                                                       f"_{self.type}_{self.date}_{self.version}.png")
        # self.predict_show_plotly_path = self.get_config('path', 'predict_show_plotly_path').replace(".png",
        #                                                                                             f"_{self.type}_{self.date}_{self.version}.png")

    def get_config(self, section, key):
        """优先命令行参数，次之yaml，最后None"""
        arg_key = f"{section}_{key}"
        if self.args is not None:
            # 支持dict或Namespace两种方式
            if isinstance(self.args, dict):
                if arg_key in self.args and self.args[arg_key] is not None:
                    return self.args[arg_key]
            else:  # Namespace
                if hasattr(self.args, arg_key) and getattr(self.args, arg_key) is not None:
                    return getattr(self.args, arg_key)

        # YAML配置
        if section in self._configs and key in self._configs[section]:
            return self._configs[section][key]

        return None

    def print_attribute(self):
        log = get_logger()
        log.info(f"{'=' * 25} start print config {'=' * 25}")
        for attr, value in self.__dict__.items():
            if not attr.startswith('_'):  # 不打印私有属性
                log.info(f"{attr}: {value}")
        log.info(f"{'=' * 25} end print config {'=' * 25}")