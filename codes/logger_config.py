import logging as logger
import os


def log(func):
    def wrapper(*args, **kwargs):
        logger.debug(f"Before executing {func.__name__}: {args}, {kwargs}")
        result = func(*args, **kwargs)
        logger.debug(f"After executing {func.__name__}: {result}")
        return result

    return wrapper


class FindfontFilter(logger.Filter):
    def filter(self, record):
        return "DEBUG findfont" not in record.getMessage()


class LoggerConfig():
    def __init__(self, log_path):
        self.log_path = log_path
        self.mylog = logger.getLogger("myapp")
        self.mylog.setLevel(logger.DEBUG)

        # 创建一个文件处理程序，将日志消息保存到 myapp.log 文件中
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        file_handler = logger.FileHandler(self.log_path)
        file_handler.setLevel(logger.DEBUG)
        # 添加过滤器以过滤掉带有 "DEBUG findfont" 的日志消息
        file_handler.addFilter(FindfontFilter())

        # 创建一个格式化程序，将日志消息格式化为 "[%(levelname)s] %(asctime)s %(message)s"
        formatter = logger.Formatter("[%(levelname)s] %(asctime)s %(message)s")

        # 将格式化程序添加到文件处理程序中
        file_handler.setFormatter(formatter)
        # 将文件处理程序添加到日志记录器中
        self.mylog.addHandler(file_handler)

        # 创建一个控制台处理程序，将日志消息输出到控制台
        console_handler = logger.StreamHandler()
        console_handler.setLevel(logger.DEBUG)
        console_handler.setFormatter(formatter)

        # 将文件处理程序添加到日志记录器中
        self.mylog.addHandler(console_handler)

def get_logger():
    return logger.getLogger("myapp")