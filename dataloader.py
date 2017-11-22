import os
import pickle
import time
from random import shuffle

import numpy as np
import pandas as pd

import features
from settings import PROJECT_ROOT
from preprocess import Preprocessor


class DataLoader():
    def __init__(self,
                 drum_list_path="dataset/audio_list.csv",
                 label_path="dataset/labels.pkl",
                 feature_names=['mfcc'],
                 batch_size=32
                 preprocess_args=None,
                 val_set_number=0,
                 is_training=True,
                 ):
        '''
        :param drum_list_path: 'dataset/audio_list.csv'
        :param label_path: 'dataset/labels.pkl'
        :param feature_names: list of features to use
        :param batch_size:
        :param preprocess_args:
        :param val_set_number: validation number for Cross-validation
        :param is_training: training / validation mode
        '''
        self.drum_list_path = drum_list_path
        self.metadata_df = pd.read_csv(self.drum_list_path)
        self.batch_size = batch_size
        self.feature_names = feature_names
        self.labels = pickle.load(open(label_path, 'rb'))
        self.val_set_number = val_set_number
        self.is_training = is_training
        self.dataset = None
        self.create_batches()  # Get Train or Validation
        assert self.dataset is not None

        if preprocess_args is not None:
            preprocessor = Preprocessor(self.dataset)
            self.dataset = preprocessor.run(args, is_training)

        self.batch_gen = self.batch_generator()


    def create_batches(self):
        '''

        :return:
        '''
        data_dir = os.path.join(PROJECT_ROOT, 'dataset/')
        all_tids = self.metadata_df.FileName.tolist()

        if self.is_training:
            self.metadata_df = self.metadata_df[self.metadata_df['set'] != self.val_set_number]
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['set'] == self.val_set_number]

        for idx, feature_name in enumerate(self.feature_names):
            features.maybe_create_data_file(feature_name, all_tids)
            st = time.time()
            print("Loading data file {}.pkl ...".format(feature_name))
            data_file_path = os.path.join(data_dir, '{}.pkl'.format(feature_name))
            print("Loading is done. Took {} seconds.".format(time.time() - st))
            if idx == 0:
                self.dataset = np.load(data_file_path)
            else:  # Stack
                self.dataset = np.concatenate(
                    (self.dataset, np.load(data_file_path)),
                    axis=1)

        self.dataset = self.dataset[self.metadata_df.index]
        self.metadata_df.reset_index(drop=True, inplace=True)

        self.num_batch = int(len(self.metadata_df) / self.batch_size)
        self.pointer = 0


    def batch_generator(self):

        while True:
            if self.pointer % self.num_batch == 0:  # Shuffle every epoch
                ind_list = [i for i in range(len(self.metadata_df))]
                shuffle(ind_list)
                self.dataset = self.dataset[ind_list]
                self.metadata_df = self.metadata_df.iloc[ind_list, :]
                self.metadata_df.reset_index(drop=True, inplace=True)

            start_pos = self.pointer * self.batch_size
            batch_meta_df = self.metadata_df.iloc[start_pos:(start_pos+self.batch_size)]
            batch_track_ids = batch_meta_df['FileName']

            self.pointer = (self.pointer + 1) % (self.num_batch)

            yield (np.expand_dims(self.dataset[batch_track_ids.index], axis=3),  # (B, H, W, C=1)
                   self.get_labels(batch_track_ids),
                   batch_track_ids.values)

    def next_batch(self):
        '''

        :return: feature array, label array (one-hot encoded)
        '''
        return next(self.batch_gen)


    def reset_pointer(self):
        self.pointer = 0


    def get_labels(self, name_list):
        # get labels from label dictionary
        # [[list of hihats],[list of kicks], [list of snares],
        #   [list of hihats],[list of kicks], [list of snares]],...
        # print("name_list", name_list)
        labels = []
        for x in name_list:
            labels.append(self.labels[x])
        labels = np.array(labels)
        return labels


if __name__ == "__main__":
    # for test

    training_loader = dataLoader('dataset/audio_list.csv', 'dataset/labels.pkl', 32, val_set_number = 0, is_training=True)
    valid_loader = dataLoader('dataset/audio_list.csv', 'dataset/labels.pkl', 32, val_set_number = 0, is_train_mode=True)


    for _ in range(training_loader.num_batch):
        track_features, label_onehot = training_loader.next_batch()
        print(len(track_features), len(label_onehot))


    for _ in range(valid_loader.num_batch):
        track_features, label_onehot = valid_loader.next_batch()






