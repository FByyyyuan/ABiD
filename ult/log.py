import logging
import sys

def levelize(levelname):
    if isinstance(levelname, str):
        return logging.getLevelName(levelname)
    else:
        return levelname  # already level

class ColorFormatter(logging.Formatter):
    color_dic = {
        "DEBUG": 37,  # white
        "INFO": 36,  # cyan
        "WARNING": 33,  # yellow
        "ERROR": 31,  # red
        "CRITICAL": 41,  # white on red bg
    }

    def format(self, record):
        color = self.color_dic.get(record.levelname, 37)  # default white
        record.levelname = "\033[{}m{}\033[0m".format(color, record.levelname)
        return logging.Formatter.format(self, record)

class Logger(logging.Logger):
    NAME = "SingletonLogger"

    @classmethod
    def get(cls, file_path=None, level="INFO", colorize=True, track_code=False):
        logging.setLoggerClass(cls)
        logger = logging.getLogger(cls.NAME)
        logging.setLoggerClass(logging.Logger)  # restore
        logger.setLevel(level)
        if logger.hasHandlers():
            if len(logger.handlers) == 2:
                return logger

            logger.handlers.clear()
        log_format = "%(levelname)s %(asctime)s | %(message)s"
        if track_code:
            log_format = (
                "%(levelname)s::%(asctime)s | [%(filename)s] [%(funcName)s:%(lineno)d] "
                "%(message)s"
            )
        date_format = "%m/%d %H:%M:%S"
        if colorize:
            formatter = ColorFormatter(log_format, date_format)
        else:
            formatter = logging.Formatter(log_format, date_format)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.propagate = False
        return logger

    def nofmt(self, msg, *args, level="INFO", **kwargs):
        level = levelize(level)
        formatters = self.remove_formats()
        super().log(level, msg, *args, **kwargs)
        self.set_formats(formatters)

    def remove_formats(self):
        formatters = []
        for handler in self.handlers:
            formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))
        return formatters

    def set_formats(self, formatters):
        for handler, formatter in zip(self.handlers, formatters):
            handler.setFormatter(formatter)

    def set_file_handler(self, file_path):
        file_handler = logging.FileHandler(file_path)
        formatter = self.handlers[0].formatter
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)
