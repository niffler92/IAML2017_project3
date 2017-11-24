import time
import os
import argparse
# from pathlib import Path
import json

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score


from dataloader import DataLoader
import models
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
log = utils.get_logger("Train", utils.log_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    # Common Arguments
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--features", default=["mfcc", "melspectrogram", "rmse"], nargs="+")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--checkpoint_path", default="", type=str,
                        help="Checkpoint can be used to load pretrained weight")
    parser.add_argument("--val_set_number", default=0, type=int)

    # Preprocessing
    parser.add_argument("--gamma_mel", default=0.2, type=float)
    parser.add_argument("--norm_type", default=2, type=int)
    parser.add_argument("--target_height", default=8, type=int)

    # Augmentation
    parser.add_argument("--max_noise", default=1.0, type=float)

    # Train
    parser.add_argument("--train_dir", default="train_dir", type=str)
    parser.add_argument("--step_save_summaries", default=10, type=int)
    parser.add_argument("--no_save_ckpt", default=False, type=bool)
    parser.add_argument("--max_epochs", default=500, type=int)

    # Train Parameters
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--momentum", default=0.97, type=float, help="usually in (0.9, 0.95, 0.97, 0.99)")
    parser.add_argument("--dropout", default=0.5, type=float)


def main(args):
    with tf.device("/gpu:0"):  # Is there any better way?
        train_dataloader = DataLoader(batch_size=args.batch_size, args=args, is_training=True)

    model = utils.find_class_by_name([models], args.model)(args)
    model.build_graph(is_training=tf.constant(True, dtype=tf.bool))

    train(model, train_dataloader, args)




def train(model, train_dataloader, valid_dataloader, args):
    """
    Args:
        args (dict)
    """

    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False,
        allow_soft_placement=True)
    )

    saver = _set_saver(session, args)
    summary_op = _create_train_summaries(model)
    summary_writer = tf.summary.FileWriter(args['train_dir'], session.graph)

    log.info("Training start")
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    log.info("Number of total parameters : {}".format(total_params))
    total_batch = train_dataloader.num_batch

    train_loss_best = 1e10
    train_acc_best = -1e10
    valid_loss_best = 1e10
    valid_acc_best = -1e10
    epoch_best = 0

    for epoch in range(args['max_epochs']):
        train_loss = 0
        train_acc = 0
        st = time.time()
        y_preds = []
        y_trues = []

        for _ in range(total_batch):
            batch_x, batch_y, track_ids = train_dataloader.next_batch()
            _, loss_, acc_, step, summary, y_pred, y_true = session.run(
                [model.train_op, model.loss, model.accuracy_op, model.global_step, summary_op,
                 model.y_pred, model.y_true], feed_dict={model.x: batch_x, model.y: batch_y})

            train_loss += loss_ / total_batch
            train_acc += acc_ / total_batch
            y_preds += y_pred.ravel().tolist()  # (1, 600)
            y_trues += y_true.ravel().tolist()

            if step % args['step_save_summaries'] == 0:
                summary_writer.add_summary(summary, global_step=step)

        valid_acc, valid_loss = valid_full(step, total_batch, summary_writer, model, valid_dataloader, session, args)

        if not args['no_save_ckpt']:
            if valid_acc > valid_acc_best:
                saver.save(session,
                           os.path.join(args['train_dir'], args['model']) + "-bs{}".format(args['batch_size']))
                           # global_step=model.global_step)  # no step

                train_loss_best = train_loss
                train_acc_best = train_acc
                valid_loss_best = valid_loss
                valid_acc_best = valid_acc
                epoch_best = epoch


        elapsed_time = time.time() - st
        real_epoch = int(step / total_batch)
        print("#######################TRAIN#########################")
        log.info("Step: {:5d} | Epoch: {:3d} | Elapsed time: {:3.2f} | "
                 "Epoch loss: {:.5f} | Epoch accuracy: {:.5f}".format(
                    int(step), real_epoch, elapsed_time, train_loss, train_acc))

        # TODO: Report F1, precision, recall by Instrument and Overall
        #for line in classification_report(y_trues, y_preds).split("\n"):
        #    print(line)
        #print(confusion_matrix(y_trues, y_preds))

    log.info("Training Finished!")
    session.close()
    summary_writer.close()

    return train_loss_best, train_acc_best, valid_loss_best, valid_acc_best, epoch_best


def valid_full(step, total_batch, summary_writer, model, valid_dataloader, session, args):
    """
    Args:
        args (dict)
    """
    avg_loss = 0
    avg_acc = 0
    st = time.time()
    y_preds = []
    y_trues = []

    for _ in range(total_batch):
        batch_x, batch_y, track_ids = valid_dataloader.next_batch()
        loss_, acc_, step, y_pred, y_true = session.run(
            [model.loss, model.accuracy_op, model.global_step,
             model.y_pred, model.y_true], feed_dict={model.x: batch_x, model.y: batch_y})


        avg_loss += loss_ / total_batch
        avg_acc += acc_ / total_batch
        # y_pred (3, 200)
        y_preds += y_pred.ravel().tolist()
        y_trues += y_true.ravel().tolist()


    summary_op = _create_valid_summaries(avg_loss, avg_acc, y_pred, y_true)
    summary = session.run(summary_op)

    if step % args['step_save_summaries'] == 0:
        summary_writer.add_summary(summary, global_step=step)

    elapsed_time = time.time() - st
    real_epoch = int(step / total_batch)
    print("#######################VALID#########################")
    log.info("Step: {:5d} | Epoch: {:3d} | Elapsed time: {:3.2f} | "
             "Valid loss: {:.5f} | Valid accuracy: {:.5f}".format(
                int(step), real_epoch, elapsed_time, avg_loss, avg_acc))

    #for line in classification_report(y_trues, y_preds).split("\n"):
    #    print(line)
    #print(confusion_matrix(y_trues, y_preds))

    return avg_acc, avg_loss


def _create_train_summaries(model):
    with tf.name_scope("summaries/train"):
        #precision = tf.metrics.precision(tf.reshape(model.y_true, [-1]), tf.reshape(model.y_pred, [-1]))[0]
        #recall = tf.metrics.recall(tf.reshape(model.y_true, [-1]), tf.reshape(model.y_pred, [-1]))[0]
        #print(recall.shape)
        #f1_score = 2 * precision * recall / (precision + recall)
        summaries = tf.summary.merge([
            tf.summary.scalar("loss", model.loss),
            tf.summary.histogram("histogram_loss", model.loss),
            tf.summary.scalar("accuracy", model.accuracy_op)
            #tf.summary.scalar("precision", precision),
            #tf.summary.scalar("recall", recall),
            #tf.summary.scalar("f1_score", f1_score)
        ])

    return summaries


def _create_valid_summaries(loss, acc, y_preds, y_trues):
    with tf.name_scope("summaries/valid"):
        summaries = tf.summary.merge([
            tf.summary.scalar("loss", tf.constant(loss)),
            tf.summary.histogram("histogram_loss", tf.constant(loss)),
            tf.summary.scalar("accuracy", tf.constant(acc)),
            #tf.summary.scalar("F1_score", tf.constant(
            #    utils.calculate_average_F1_score(np.asarray(y_trues, dtype=int), np.asarray(y_preds, dtype=int))))
        ])

    return summaries


def _set_saver(session, args):
    saver = tf.train.Saver(tf.global_variables())
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    if args['checkpoint_path'] is not "":
        # if Path(args.checkpoint_path).is_dir():
        if os.path.isdir(args['checkpoint_path']): # for python 2.7 compatibility (need to be confirmed)
            old_checkpoint_path = args['checkpoint_path']
            args['checkpoint_path'] = tf.train.latest_checkpoint(args['checkpoint_path'])
            log.info("Update checkpoint_path: {} -> {}".format(
                old_checkpoint_path, args['checkpoint_path'])
            )
        saver.restore(session, args['checkpoint_path'])
        log.info("Restore from {}".format(args['checkpoint_path']))
    else:
        log.info("No designated checkpoint path. Initializing weights randomly.")

    return saver


if __name__ == '__main__':
    args = parser.parse_args()
    utils.save_args(args)

    print(args)
    main(args)
