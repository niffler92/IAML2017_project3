import time
import os
import argparse
# from pathlib import Path
import json

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


from dataloader import DataLoader
import models
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
log = utils.get_logger("Train", utils.log_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    # Common Arguments
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--features", default=["mfcc"], nargs="+")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--checkpoint_path", default="", type=str,
                        help="Checkpoint can be used to load pretrained weight")
    parser.add_argument("--val_set_number", default=0, type=int)

    # Preprocessing
    parser.add_argument("--gamma_mel", default=0.2, type=float)
    parser.add_argument("--norm_type", default=2, type=int)
    parser.add_argument("--target_height", default=8, type=int)

    # Train
    parser.add_argument("--train_dir", default="train_dir", type=str)
    parser.add_argument("--step_save_summaries", default=10, type=int)
    parser.add_argument("--step_save_checkpoint", default=40, type=int)
    parser.add_argument("--max_ckpt_to_keep", default=3, type=int)
    parser.add_argument("--max_epochs", default=500, type=int)

    # Train Parameters
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--momentum", default=0.97, type=float, help="usually in (0.9, 0.95, 0.97, 0.99)")
    parser.add_argument("--dropout", default=0.5, type=float)


def main(args):
    with tf.device("/gpu:0"):  # Is there any better way?
        train_dataloader = DataLoader(feature_names=args.features,
                                      batch_size=args.batch_size,
                                      preprocess_args=args,
                                      val_set_number=args.val_set_number,
                                      is_training=True)

    model = utils.find_class_by_name([models], args.model)(args)
    model.build_graph(is_training=tf.constant(True, dtype=tf.bool))

    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False,
        allow_soft_placement=True)
    )
    train(model, train_dataloader, session, args)


def train(model, dataloader, session, args):
    saver = _set_saver(session, args)
    summary_op = _create_summaries(model)
    summary_writer = tf.summary.FileWriter(args.train_dir, session.graph)

    log.info("Training start")
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    log.info("Number of total parameters : {}".format(total_params))
    total_batch = dataloader.num_batch
    for epoch in range(args.max_epochs):
        avg_loss = 0
        avg_acc = 0
        st = time.time()
        y_preds = []
        y_trues = []

        for _ in range(total_batch):
            batch_x, batch_y, track_ids = dataloader.next_batch()
            _, loss_, acc_, step, summary, y_pred, y_true = session.run(
                [model.train_op, model.loss, model.accuracy_op, model.global_step, summary_op,
                 model.y_pred, model.y_true], feed_dict={model.x: batch_x, model.y: batch_y})

            avg_loss += loss_ / total_batch
            avg_acc += acc_ / total_batch
            y_preds += y_pred.ravel().tolist()
            y_trues += y_true.ravel().tolist()

            if step % args.step_save_summaries == 0:
                summary_writer.add_summary(summary, global_step=step)
                # log.info("[Step:{}] Save training summary: {}".format(step, args.train_dir))
            if step % args.step_save_checkpoint == 0:
                saver.save(session,
                           os.path.join(args.train_dir, args.model) + "-bs{}".format(args.batch_size),
                           global_step=model.global_step)
                # log.info("[Step:{}] Save checkpoint: {}".format(step, args.train_dir))

        elapsed_time = time.time() - st
        real_epoch = int(step / total_batch)
        log.info("Step: {:5d} | Epoch: {:3d} | Elapsed time: {:3.2f} | "
                 "Epoch loss: {:.5f} | Epoch accuracy: {:.5f}".format(
                    int(step), real_epoch, elapsed_time, avg_loss, avg_acc))

        for line in classification_report(y_trues, y_preds).split("\n"):
            print(line)
        print(confusion_matrix(y_trues, y_preds))

    log.info("Training Finished!")
    session.close()
    summary_writer.close()


def _create_summaries(model):
    with tf.name_scope("summaries/train"):
        tf.summary.scalar("loss", model.loss)
        tf.summary.histogram("histogram_loss", model.loss)
        tf.summary.scalar("accuracy", model.accuracy_op)

    summaries = tf.summary.merge_all()
    return summaries


def _set_saver(session, args):
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.max_to_keep)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    if args.checkpoint_path is not "":
        # if Path(args.checkpoint_path).is_dir():
        if os.path.isdir(args.checkpoint_path): # for python 2.7 compatibility (need to be confirmed)
            old_checkpoint_path = args.checkpoint_path
            args.checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_path)
            log.info("Update checkpoint_path: {} -> {}".format(
                old_checkpoint_path, args.checkpoint_path)
            )
        saver.restore(session, args.checkpoint_path)
        log.info("Restore from {}".format(args.checkpoint_path))
    else:
        log.info("No designated checkpoint path. Initializing weights randomly.")

    return saver


if __name__ == '__main__':
    args = parser.parse_args()
    utils.save_args(args)

    print(args)
    main(args)
