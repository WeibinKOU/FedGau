import os
import logging
import coloredlogs
import datetime as dt


class Formatter(logging.Formatter):
    """Log print formatter"""

    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, date_format=None):
        ct = self.converter(record.created)
        if date_format:
            s = ct.strftime(date_format)
        else:
            # t = ct.strftime("%Y-%m-%d %H:%M:%S")
            t = ct.strftime("%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s


class Logger(object):

    logger = None

    def __init__(self, log_name, log_dir, level='debug', shell=True):

        self.logger = logging.getLogger(log_name)

        if level == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif level == 'warning':
            self.logger.setLevel(logging.WARNING)
        elif level == 'error':
            self.logger.setLevel(logging.ERROR)
        else:
            self.logger.setLevel(logging.INFO)

        # fmt = "%(asctime)-15s %(levelname)s %(process)d %(message)s"
        fmt = "%(asctime)-15s %(levelname)s %(message)s"
        formatter = Formatter(fmt)

        if shell:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

        fh = logging.FileHandler(os.path.join(log_dir, log_name + '.log'), mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        coloredlogs.install(level=level, logger=self.logger, fmt=fmt)

    def info(self, *msgs):
        self.logger.info(" ".join((str(s) for s in msgs)))

    def warning(self, *msgs):
        self.logger.warning(" ".join((str(s) for s in msgs)))

    def debug(self, *msgs):
        self.logger.debug(" ".join((str(s) for s in msgs)))

    def error(self, *msgs):
        self.logger.error(" ".join((str(s) for s in msgs)))
