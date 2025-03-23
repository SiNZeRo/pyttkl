import logging
import os
import sys
import multiprocessing
from datetime import datetime, timedelta
import time
import fcntl
from ._args import make_args, run_cmds, remove_func, GETARGS, make_sub_cmd
from . import _args as etargs

logger = logging.getLogger(__name__)

TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.TRACE = TRACE_LEVEL_NUM

__global_ctx = {}

def get_ctx():
    global __global_ctx
    return __global_ctx

def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)
        logging.Logger.trace = trace

def logging_trace(logger, message, *args, **kws):
    '''
    Log a message with level TRACE on the root logger.
    '''
    if logger.isEnabledFor(TRACE_LEVEL_NUM):
        logger.trace(message, *args, **kws)

if True:
    logging.Logger.trace = trace

def get_logger(name):
    return logging.getLogger(name)


def init_logging(level, filename=None, not_debugs=['matplotlib', 'diff'], force=False):
    format_str = "|%(asctime)s.%(msecs)03d| %(name)-10.10s | %(filename)10.10s:%(lineno)-4d | %(funcName)-20.20s | %(levelname)-5.5s | %(message)s"

    class CustomFormatter(logging.Formatter):
        reset = "\033[1;0m"

        FORMATS = {
            logging.TRACE: "\033[1;2m" + format_str + reset,
            logging.DEBUG: "\033[1;0m" + format_str + reset,
            logging.INFO: "\033[1;1m" + format_str + reset,
            logging.WARNING: "\033[1;33m" + format_str + reset,
            logging.ERROR: "\033[1;31m" + format_str + reset,
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, "%Y%m%dT%H:%M:%S")
            tmp = formatter.format(record)
            return tmp

    if '--ltrace' in sys.argv:
        level = 'trace'
        sys.argv.remove('--ltrace')
    elif '--ldebug' in sys.argv:
        level = 'debug'
        sys.argv.remove('--ldebug')
    elif '--linfo' in sys.argv:
        level = 'info'
        sys.argv.remove('--linfo')

    if not get_ctx().get('__logging_init__') or force:
        get_ctx()['__logging_init__'] = True
    else:
        return

    has_color = tty = sys.stdout.isatty()

    rootLogger = logging.getLogger()
    if level == 'debug':
        rootLogger.setLevel(logging.DEBUG)
        for mod in not_debugs:
            logging.getLogger(mod).setLevel(logging.INFO)
    elif level == 'info':
        rootLogger.setLevel(logging.INFO)
    elif level == 'trace':
        rootLogger.setLevel(TRACE_LEVEL_NUM)

    if has_color:
        logFormatter = CustomFormatter()

    else:
        logFormatter = logging.Formatter(format_str, "%Y%m%dT%H:%M:%S")

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    consoleHandler.setLevel(TRACE_LEVEL_NUM)

    logger.trace(f"logging init, level={level}")

    # logging.info(f"logging init, level={level}")
