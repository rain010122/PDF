# config.py
import argparse
import yaml
from datetime import datetime
from logger_config import *

# ===== 1. 集中式默认值表（维护就方便）=====
PARAM_DEFAULTS = {
    # ---------- common ----------
    'common_random_seed': 2021,
    'common_is_training': 1,
    'common_data': 'ETTm1',
    'common_features': 'M',
    'common_target': 'OT',
    'common_freq': 'h',
    'common_seq_len': 96,
    'common_label_len': 48,
    'common_pred_len': 96,
    'common_model_id': 'test',
    'common_model': 'Autoformer',
    # ---------- path ----------
    'path_root_path': "../data/ETT/",
    'path_data_path': "ETTh1.csv",
    'path_checkpoints': "../checkpoints/",
    'path_log_path': "../data/logs/log.log",
    # ---------- model_params ----------
    'model_params_fc_dropout': 0.0,
    'model_params_head_dropout': 0.0,
    'model_params_add': False,
    'model_params_wo_conv': False,
    'model_params_serial_conv': False,
    'model_params_kernel_list': [3, 7, 9],
    'model_params_patch_len': [16],
    'model_params_period': [24, 12],
    'model_params_stride': None,
    'model_params_padding_patch': 'end',
    'model_params_revin': 1,
    'model_params_affine': 0,
    'model_params_subtract_last': 0,
    'model_params_decomposition': 0,
    'model_params_kernel_size': 25,
    'model_params_individual': 0,
    'model_params_embed_type': 0,
    'model_params_enc_in': 7,
    'model_params_dec_in': 7,
    'model_params_c_out': 7,
    'model_params_d_model': 512,
    'model_params_n_heads': 8,
    'model_params_e_layers': 2,
    'model_params_d_layers': 1,
    'model_params_d_ff': 2048,
    'model_params_moving_avg': 25,
    'model_params_factor': 1,
    'model_params_distil': True,
    'model_params_dropout': 0.05,
    'model_params_attn_dropout': 0.05,
    'model_params_embed': "timeF",
    'model_params_activation': "gelu",
    'model_params_output_attention': False,
    'model_params_do_predict': False,
    # ---------- train_params ----------
    'train_params_num_workers': 0,   # 开发为主进程加载
    'train_params_itr': 2,
    'train_params_train_epochs': 100,
    'train_params_batch_size': 16,
    'train_params_patience': 100,
    'train_params_learning_rate': 0.0001,
    'train_params_des': "test",
    'train_params_loss': "mse",
    'train_params_lradj': "type3",
    'train_params_pct_start': 0.3,
    'train_params_use_amp': False,
    # ---------- gpu ----------
    'gpu_use_gpu': True,
    'gpu_gpu': 0,
    'gpu_use_multi_gpu': False,
    'gpu_devices': '0,1,2,3',
    'gpu_test_flop': False,
}


class Config(object):
    def __init__(self, config_path=None, args=None):
        """
        支持三层优先级: 命令行args > yaml > default
        args: 为dict或argparse.Namespace
        """
        self.args = args if args is not None else None

        # 1. 读取yaml
        self._configs = {}
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._configs = yaml.safe_load(f)
        # 2. 时间相关附加信息
        self.date = datetime.now().strftime('%Y%m%d')
        self.version = datetime.now().strftime("%H%M%S")

        # ==== common ====
        self.random_seed = self.get_config('common', 'random_seed')
        self.is_training = self.get_config('common', 'is_training')
        self.data = self.get_config('common', 'data')
        self.features = self.get_config('common', 'features')
        self.target = self.get_config('common', 'target')
        self.freq = self.get_config('common', 'freq')
        self.seq_len = self.get_config('common', 'seq_len')
        self.label_len = self.get_config('common', 'label_len')
        self.pred_len = self.get_config('common', 'pred_len')
        self.model_id = f"{self.data}_{self.seq_len}_{self.pred_len}"
        self.model = self.get_config('common', 'model')
        # ==== path ====
        self.root_path = self.get_config('path', 'root_path')
        self.data_path = self.get_config('path', 'data_path')
        self.checkpoints = self.get_config('path', 'checkpoints')
        log_path_base = self.get_config('path', 'log_path')
        # 支持日志自动append时间戳（不想带时间戳可直接self.log_path=xxx）
        self.log_path = log_path_base.replace(".log", f"{self.model_id}_{self.model}_{self.date}_{self.version}.log")
        # ==== model_params ====
        mp = 'model_params'
        self.fc_dropout = self.get_config(mp, 'fc_dropout')
        self.head_dropout = self.get_config(mp, 'head_dropout')
        self.add = self.get_config(mp, 'add')
        self.wo_conv = self.get_config(mp, 'wo_conv')
        self.serial_conv = self.get_config(mp, 'serial_conv')
        self.kernel_list = self.get_config(mp, 'kernel_list')
        self.patch_len = self.get_config(mp, 'patch_len')
        self.period = self.get_config(mp, 'period')
        self.stride = self.get_config(mp, 'stride')
        self.padding_patch = self.get_config(mp, 'padding_patch')
        self.revin = self.get_config(mp, 'revin')
        self.affine = self.get_config(mp, 'affine')
        self.subtract_last = self.get_config(mp, 'subtract_last')
        self.decomposition = self.get_config(mp, 'decomposition')
        self.kernel_size = self.get_config(mp, 'kernel_size')
        self.individual = self.get_config(mp, 'individual')
        self.embed_type = self.get_config(mp, 'embed_type')
        self.enc_in = self.get_config(mp, 'enc_in')
        self.dec_in = self.get_config(mp, 'dec_in')
        self.c_out = self.get_config(mp, 'c_out')
        self.d_model = self.get_config(mp, 'd_model')
        self.n_heads = self.get_config(mp, 'n_heads')
        self.e_layers = self.get_config(mp, 'e_layers')
        self.d_layers = self.get_config(mp, 'd_layers')
        self.d_ff = self.get_config(mp, 'd_ff')
        self.moving_avg = self.get_config(mp, 'moving_avg')
        self.factor = self.get_config(mp, 'factor')
        self.distil = self.get_config(mp, 'distil')
        self.dropout = self.get_config(mp, 'dropout')
        self.attn_dropout = self.get_config(mp, 'attn_dropout')
        self.embed = self.get_config(mp, 'embed')
        self.activation = self.get_config(mp, 'activation')
        self.output_attention = self.get_config(mp, 'output_attention')
        self.do_predict = self.get_config(mp, 'do_predict')
        # ==== train_params ====
        tp = 'train_params'
        self.num_workers = self.get_config(tp, 'num_workers')
        self.itr = self.get_config(tp, 'itr')
        self.train_epochs = self.get_config(tp, 'train_epochs')
        self.batch_size = self.get_config(tp, 'batch_size')
        self.patience = self.get_config(tp, 'patience')
        self.learning_rate = self.get_config(tp, 'learning_rate')
        self.des = self.get_config(tp, 'des')
        self.loss = self.get_config(tp, 'loss')
        self.lradj = self.get_config(tp, 'lradj')
        self.pct_start = self.get_config(tp, 'pct_start')
        self.use_amp = self.get_config(tp, 'use_amp')
        # ==== gpu ====
        gp = 'gpu'
        self.use_gpu = self.get_config(gp, 'use_gpu')
        self.gpu = self.get_config(gp, 'gpu')
        self.use_multi_gpu = self.get_config(gp, 'use_multi_gpu')
        self.devices = self.get_config(gp, 'devices')
        self.test_flop = self.get_config(gp, 'test_flop')

    def get_config(self, section, key):
        """优先级: 命令行 > yaml > 默认值 > None（极少情况）"""
        arg_key = f"{section}_{key}"
        # 从命令行args优先
        if self.args is not None:
            if isinstance(self.args, dict):
                if arg_key in self.args and self.args[arg_key] is not None:
                    return self.args[arg_key]
            else:
                if hasattr(self.args, arg_key) and getattr(self.args, arg_key) is not None:
                    return getattr(self.args, arg_key)
        # 再yaml
        if section in self._configs and key in self._configs[section]:
            return self._configs[section][key]
        # 最后default
        if arg_key in PARAM_DEFAULTS:
            return PARAM_DEFAULTS[arg_key]
        return None

    def as_dict(self):
        # 只包含所有非私有成员变量（且只一份）
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    def print_attribute(self):
        print("=" * 25, "start config", "=" * 25)
        for k, v in self.as_dict().items():
            print(f"{k}: {v}")
        print("=" * 25, "end config", "=" * 25)


# ======== 推荐的获取config接口 ========
# def get_config_from_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_file', type=str, default=None, help='Path to yaml config')
#     # 你可以随意扩展更多命令行参数（注意arg_key命名），如：
#     # parser.add_argument('--common_seq_len', type=int, default=None)
#     # parser.add_argument('--common_model', type=str, default=None)
#     args, _ = parser.parse_known_args()
#     cfg = Config(config_path=args.config_file, args=args)
#     return cfg

# if __name__ == "__main__":
#     cfg = get_config_from_args()
#     cfg.print_attribute()