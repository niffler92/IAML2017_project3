import os
import argparse
import glob

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import utils
import models
from dataloader import DataLoader
from train import *
from settings import PROJECT_ROOT
from params import param_batch as param


log = utils.get_logger("Ensemble", None)

def ensemble(args):
    """Validate Ensemble model with full validation set.... (Hmm)
    """

    max_models = utils.get_arg(args, "max_models")
    criterion = utils.get_arg(args, "criterion")
    method = utils.get_arg(args, "method")
    filename = utils.get_arg(args, "filename")

    log.info("Starting ensemble of {} models with existing ckpt from {}".format(max_models, filename))
    log.info("Criterion: {}, Method: {}".format(criterion, method))

    pass_empty = 0

    valid_dataloader = DataLoader(total_validation=True)

    for i in range(max_models):
        param_dict, pass_empty = get_best_hyperparams(filename, criterion, i, pass_empty)
        valid_dataloader.reset_args(param_dict)

        tf.reset_default_graph()

        model = utils.find_class_by_name([models], param_dict['model'])(param_dict)
        model.build_graph(is_training=tf.constant(False, dtype=tf.bool))

        session = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True),
            log_device_placement=False,
            allow_soft_placement=True)
        )
        restore_session(session, param_dict)

        _, _, _, y_logit, y_true = valid_full(0, model, valid_dataloader, session, param_dict)
        session.close()

        assert y_logit.shape == (3, 87*200)
        assert y_true.shape[0] == (3, 87*200)
        y_logit = y_logit.reshape([1, -1])
        y_true = y_true.reshape([1, -1])

        if i == 0:
            y_logits = y_logit
            y_trues = y_true
        else:
            y_logits = np.concatenate((y_logits, y_logit), axis=0)
            assert y_trues == y_true, "Must return same true values"

    if criterion == "average":
        y_logits = np.mean(y_logits, axis=0)
        y_preds = np.greater(y_logits, 0).astype(int)
    elif criterion == "vote":
        y_logits = np.greater(y_logits, 0).astype(int)
        one_counts = np.count_nonzero(y_logits, axis=0)
        # Warning: Predicts Tie as 1 if (max_models % 2 == 0)
        y_preds = np.greater_equal(one_counts, max_models - one_counts).astype(int)
    else:
        raise ValueError("Invalid criterion type: {}".format(criterion))

    print("#############ENSEMBLE RESULT##############")
    for line in classification_report(y_trues, y_preds).split("\n"):
        print(line)
    print(confusion_matrix(y_trues, y_preds))
    print("F1_score: {}".format(f1_score(y_trues, y_preds)))



def get_best_hyperparams(filename, criterion, i, pass_empty):
    """Returns hyperparameters of ith best result in result.txt where ckpt exists
    Args:
        criterion (str): valid_f1 or valid_acc
        i (int): ith best hyperparam to retrieve

    Returns:
        best_params (dict)
    """

    df_result = pd.read_csv(os.path.join(PROJECT_ROOT, filename), header=0, delimiter="\t")
    df_result = df_result[df_result.no_save_ckpt == False].sort_values(by=criterion, ascending=False)

    while True:
        best_params = df_result.iloc[i+pass_empty].to_dict()
        ckpt_path = os.path.join(PROJECT_ROOT, best_params['train_dir'], best_params['tag_label'], "*{}*".format(best_params['unique_key']))
        ckpt_list = glob.glob(ckpt_path)
        if len(ckpt_list) == 3:
            break;
        else:
            log.warning("Result with unique key({}) doesn't have ckpt files in the directory. "
                        "It has been trained to save ckpt. Proceeding to next best model...")
            pass_empty += 1

    return best_params, pass_empty


def restore_session(session, param_dict):
    saver = tf.train.Saver(tf.global_variables())
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    ckpt_path = os.path.join(PROJECT_ROOT, param_dict['train_dir'], param_dict['tag_label'], "*{}*".format(param_dict['unique_key']))
    ckpt_list = glob.glob(ckpt_path)
    data_path = [ckpt for ckpt in ckpt_list if '.data' in ckpt][0]

    try:
        log.info("Restoring from {}".format(data_path))
        saver.restore(session, data_path)
    except:
        raise Exception("Something is wrong with ckpt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--max_models", default=3, type=int)
    parser.add_argument("--criterion", default="valid_f1", type=str, choices=["valid_f1", "valid_acc"])
    parser.add_argument("--method", default="average", type=str, choices=["average", "vote"])
    parser.add_argument("--filename", default="log/batch_log/result_experiment.txt", type=str)

    args = parser.parse_args()
    ensemble(args)
