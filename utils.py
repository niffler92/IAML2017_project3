import json
import logging
from datetime import datetime
from collections import Counter
import argparse
import getpass
import copy

import os
import settings
import numpy as np
import pandas as pd

from settings import PROJECT_ROOT


def get_logger(logger_name, log_file=None, level=logging.DEBUG):
    # "log/data-pipe-{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s")

    logger.setLevel(level)

    if log_file is not None:
        # log_file.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))

        fileHandler = logging.FileHandler(log_file, mode="w")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    return logger


# log_path = Path(settings.PROJECT_ROOT + "/log/common-{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
# for python 2.7 compatibility (need to be confirmed)
log_path = os.path.join(settings.PROJECT_ROOT, "log/common-{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

# log = get_logger("common", log_path)


def find_class_by_name(modules, name):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def save_args(args, filename="train_params.json"):
    """Saves hyperparameters in training directory
    """
    # save_path = (Path(args.train_dir) / Path(filename)).as_posix()
    # if not Path(save_path).parent.exists():
    #     Path(save_path).parent.mkdir()

    # for python 2.7 compatibility (need to be confirmed)
    save_path = os.path.join(args.train_dir, filename)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, 'w') as fp:
        json.dump(vars(args), fp)


def load_args(args, filename="train_params.json"):
    # load_path = (Path(args.train_dir) / Path(filename)).as_posix()
    load_path = os.path.join(args.train_dir, filename)  # for python 2.7 compatibility (need to be confirmed)
    with open(load_path, 'r') as fp:
        result = json.load(fp)

    return result


def get_random_param(param_list_dict):
    param_dict = dict()
    for key, val_list in param_list_dict.items():
        len_param = len(val_list)
        index = np.random.randint(len_param)
        val = val_list[index]
        param_dict[key] = val

    return param_dict


#def write_param(param_dict, train_cost, train_acc, valid_cost, valid_acc, epoch):
#    filename = param_dict['train_batch_result_filename']
#
#    if not os.path.exists(filename):
#        file = open(filename, 'w')
#        for key in param_dict.keys():
#            file.write(key + '\t')
#        file.write('train_cost\t')
#        file.write('train_acc\t')
#        file.write('valid_cost\t')
#        file.write('valid_acc\t')
#        file.write('epoch\n')
#    else:
#        file = open(filename, 'a')
#
#    for key in param_dict.keys():
#        param = param_dict[key]
#        if type(param) == str:
#            file.write('%s\t' % param)
#        elif type(param) == int:
#            file.write('%d\t' % param)
#        elif type(param) == float:
#            file.write('%f\t' % param)
#        elif type(param) == list and type(param[0]) == str:
#            for p in param:
#                file.write('%s ' % p)
#            file.write('\t')
#        elif type(param) == bool:
#            file.write('%s\t' % p)
#        else:
#            raise ValueError('param type error, {}: {}'.format(key, type(param)))
#
#    file.write('%f\t' % (train_cost))
#    file.write('%f\t' % (train_acc))
#    file.write('%f\t' % (valid_cost))
#    file.write('%f\t' % (valid_acc))
#    file.write('%d\n' % (epoch))
#    file.close()
#
#
def write_param(prm_dict, **kwargs):
    """kwargs:  train_cost, train_acc, valid_cost, valid_acc, epoch
    """
    assert isinstance(prm_dict, dict)
    param_dict = copy.deepcopy(prm_dict)

    kwargs.update({"username": getpass.getuser()})
    param_dict.update(kwargs)
    filename = param_dict['train_batch_result_filename']
    filepath = os.path.join(PROJECT_ROOT, filename)

    if not os.path.exists(filepath):
        df = pd.DataFrame()  # param_dict inside X
        df = df.append(param_dict, ignore_index=True)
    else:
        df = pd.read_csv(filepath, header=0, delimiter="\t")
        df = df.append(param_dict, ignore_index=True)

    df.to_csv(filepath, sep="\t", index=False)


def calculate_average_F1_score(pred_lists, label_lists):
    # calculate average F1 score (hihat, kick, snare)
    # shape of each list is 3*200
    avg_f1_score = 0
    n = 0
    for pred_list, label_list in zip(pred_lists, label_lists):
        counts = Counter(zip(pred_list, label_list))
        tp = counts[1,1]
        fp = counts[1,0]
        fn = counts[0,1]
        try:
            precision = float(tp) / (tp+fp)
        except ZeroDivisionError:
            precision = 0

        try:
            recall = float(tp) / (fn + tp)
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2*(precision*recall / (precision+recall))
        except ZeroDivisionError:
            f1 = 0
        avg_f1_score+=f1

        n +=1
        # print(precision, recall, f1)
    avg_f1_score /= n

    return avg_f1_score


def get_arg(args, attr):
    if isinstance(args, dict):
        return args[attr]
    elif isinstance(args, argparse.Namespace):
        return getattr(args, attr)
    else:
        raise ValueError("Unknown args")
