import json
import logging
from datetime import datetime

from pathlib import Path

import settings


def get_logger(logger_name, log_file=None, level=logging.DEBUG):
    # "log/data-pipe-{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s")

    logger.setLevel(level)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fileHandler = logging.FileHandler(log_file, mode="w")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    return logger


log_path = Path(settings.PROJECT_ROOT + "/log/common-{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
# log = get_logger("common", log_path)


def find_class_by_name(modules, name):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def save_args(args, filename="train_params.json"):
    """Saves hyperparameters in training directory
    """
    save_path = (Path(args.train_dir) / Path(filename)).as_posix()
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir()

    with open(save_path, 'w') as fp:
        json.dump(vars(args), fp)


def load_args(args, filename="train_params.json"):
    load_path = (Path(args.train_dir) / Path(filename)).as_posix()
    with open(load_path, 'r') as fp:
        result = json.load(fp)

    return result
