import tensorflow as tf
import pandas as pd
import numpy as np
from dataloader import dataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
from collections import Counter
### properties
# General
# TODO : declare additional properties
# not fixed (change or add property as you like)
batch_size = 16
epoch_num = 100


# fixed
audio_list_path = 'dataset/audio_list.csv'
label_path = 'dataset/labels.pkl'
# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = False
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint/'

# Placeholder and variables
# TODO : declare placeholder and variables

# Build model
# TODO : build your model here

# Loss and optimizer
# TODO : declare loss and optimizer operation
# Train and evaluate
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()



    if is_train_mode:
        for epoch in range(epoch_num):
            val_set_number = 0
            # TODO: perform cross validation
            train_dataloader = dataLoader(drum_list_path=audio_list_path, label_path=label_path, batch_size=batch_size,
                                          val_set_number=val_set_number, is_train_mode = True)

            total_batch = train_dataloader.num_batch

            for i in range(total_batch):
                batch_x, batch_y = train_dataloader.next_batch()
                # TODO:  do some train step code here

        print('Training finished !')
        output_dir = checkpoint_path + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        saver.save(sess, output_dir)
        print('Model saved in file : %s'%output_dir)
    else:
        # skip training and restore graph for validation test
        saver.restore(sess, checkpoint_path)

    # Validation
    validation_dataloader = dataLoader(drum_list_path=audio_list_path, label_path = label_path, batch_size = batch_size, val_set_number= val_set_number, is_train_mode = False)

    average_val_cost = 0
    for i in range(validation_dataloader.num_batch):
        batch_x, batch_y = validation_dataloader.next_batch()
        # TODO : do some loss calculation here

    print('Validation loss : %f'%average_val_cost)

    # accuracy test example
    # TODO : accuracy test code. use calculate_average_F1_score

def calculate_average_F1_score(pred_lists, label_lists):
    # calculate average F1 score (hihat, kick, snare)
    # shape of each list is 3*200

    avg_f1_score = 0
    for pred_list, label_list in zip(pred_lists, label_lists):
        counts = Counter(zip(pred_list, label_list))
        tp = counts[1,1]
        fp = counts[1,0]
        fn = counts[0,1]
        try:
            precision = tp / (tp+fp)
        except ZeroDivisionError:
            precision = 0

        try:
            recall = tp / (fn + tp)
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2*(precision*recall / (precision+recall))
        except ZeroDivisionError:
            f1 = 0
        avg_f1_score+=f1

        print(precision, recall, f1)

    avg_f1_score /= 3
    return avg_f1_score


















