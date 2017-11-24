import os
import pickle
import time
from random import shuffle
import copy

import numpy as np
import pandas as pd

import features
from settings import PROJECT_ROOT
from preprocess import Preprocessor
from augmentation import Augmentation
import matplotlib.pyplot as plt
from params import param_default as param
import utils

class DataLoader():
    def __init__(self,
                 drum_list_path=os.path.join(PROJECT_ROOT, "dataset/audio_list.csv"),
                 label_path=os.path.join(PROJECT_ROOT, "dataset/labels.pkl"),
                 feature_names=None,
                 batch_size=None,
                 args=None,
                 val_set_number=0,
                 is_training=True,
                 total_validation=False):

        '''
        :param drum_list_path: 'dataset/audio_list.csv'
        :param label_path: 'dataset/labels.pkl'
        :param is_training: training / validation mode
        '''
        self.feature_names = feature_names
        self.batch_size = batch_size
        self.val_set_number = val_set_number
        self.drum_list_path = drum_list_path
        self.metadata_df_org = pd.read_csv(self.drum_list_path)
        self.total_validation = total_validation

        self.labels = pickle.load(open(label_path, 'rb'))
        self.is_training = is_training
        self.dataset_org = None

        if args is not None:
            self.reset_args(args)

    def reset_args(self, args):
        # arguments setting
        self.args = args

        if self.batch_size is None:
            self.batch_size = utils.get_arg(self.args, 'batch_size')
        if self.val_set_number is None:
            self.val_set_number = utils.get_arg(self.args, 'val_set_number')
        if self.feature_names is None:
            self.feature_names = utils.get_arg(self.args, 'feature_names')

        if self.dataset_org is None:
            self.create_dataset_org()
            assert self.dataset_org is not None

        self.dataset = copy.deepcopy(self.dataset_org)
        self.metadata_df = self.metadata_df_org.copy(deep=True)
        self.create_batches()  # Get Train/Validation from original

        preprocessor = Preprocessor(self.dataset, args, self.is_training, self.feature_names)
        self.dataset = preprocessor.run()
        for i, key in enumerate(['height', 'width', 'depth']):
            self.args[key] = self.dataset[0].shape[i]
        self.batch_gen = self.batch_generator()


    def create_dataset_org(self):
        data_dir = os.path.join(PROJECT_ROOT, 'dataset/')
        all_tids = self.metadata_df_org.FileName.tolist()

        for idx, feature_name in enumerate(self.feature_names):
            features.maybe_create_data_file(feature_name, all_tids)
            st = time.time()
            print("Loading data file {}.pkl ...".format(feature_name))
            data_file_path = os.path.join(data_dir, '{}.pkl'.format(feature_name))
            print("Loading is done. Took {} seconds.".format(time.time() - st))
            if idx == 0:
                self.dataset_org = np.load(data_file_path)
            else:  # Stack
                self.dataset_org = np.concatenate(
                    (self.dataset_org, np.load(data_file_path)),
                    axis=1)


    def create_batches(self):
        '''

        :return:
        '''
        if self.total_validation:
            pass
        elif self.is_training:
            self.metadata_df = self.metadata_df[self.metadata_df['set'] != self.val_set_number]
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['set'] == self.val_set_number]

        self.dataset = copy.deepcopy(self.dataset_org[self.metadata_df.index])
        self.metadata_df.reset_index(drop=True, inplace=True)
        self.num_batch = int((len(self.metadata_df) + self.batch_size - 1) / self.batch_size)
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

            yield (self.dataset[batch_track_ids.index],  # (B, H, W, C)
                   self.get_labels(batch_track_ids),
                   batch_track_ids.values)


    def next_batch(self):
        '''
        :return: feature array, label array (one-hot encoded)
        '''
        features, label_onehot, titles = next(self.batch_gen)
        augmentation = Augmentation(features, self.args, self.is_training)
        features = augmentation.run()

        return features, label_onehot, titles


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

    param_list_dict = param.param_list_dict()
    param_dict = utils.get_random_param(param_list_dict)

    training_loader = DataLoader(val_set_number=0, args=param_dict, batch_size=8, is_training=True)
    valid_loader = DataLoader(val_set_number=0, args=param_dict, batch_size=8, is_training=False)

    for _ in range(training_loader.num_batch):
        features, label_onehot, titles = training_loader.next_batch()

        data = np.concatenate([features[0, :, :, 0], features[0, :, :, 1], features[0, :, :, 2], features[0, :, :, 3],
                               features[0, :, :, 4], features[0, :, :, 5], features[0, :, :, 6], features[0, :, :, 7],
                               features[0, :, :, 8], features[0, :, :, 9], features[0, :, :, 10], features[0, :, :, 11],
                               features[0, :, :, 12]], axis=0)

        plt.figure(1)
        plt.imshow(np.squeeze(data))
        plt.show()

        print(len(features), len(label_onehot))


    for _ in range(valid_loader.num_batch):
        features, label_onehot, titles = valid_loader.next_batch()
        print(len(features), len(label_onehot))







