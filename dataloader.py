import numpy as np
import pandas as pd
import features
import pickle
'''

'''
class dataLoader():
    def __init__(self, drum_list_path, label_path, batch_size, val_set_number = 0, is_train_mode = True):
        '''
        :param drum_list_path: file path for track_metadata.csv
        :param label_path: label path
        :param batch_size:
        :param label_column_name: column name of label (project 1: track_genre_top, project 2: listens)
        :param is_training: training / validation mode
        '''
        self.batch_size = batch_size
        self.token_stream = []
        self.drum_list_path = drum_list_path
        self.labels = pickle.load(open(label_path, 'rb'))
        self.train = is_train_mode
        self.val_set_number = val_set_number
        self.create_batches()



    def create_batches(self):
        '''

        :return:
        '''
        self.metadata_df = pd.read_csv(self.drum_list_path)

        if self.train:
            self.metadata_df = self.metadata_df[self.metadata_df['set'] != self.val_set_number]
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['set'] == self.val_set_number]

        self.metadata_df = self.metadata_df.sample(frac=1).reset_index(drop=True)

        self.num_batch = int(len(self.metadata_df) / self.batch_size)
        self.pointer = 0

    def next_batch(self):
        '''

        :return: feature array, label array (one-hot encoded)
        '''
        self.pointer = (self.pointer + 1) % self.num_batch

        start_pos = self.pointer * self.batch_size
        meta_df = self.metadata_df.iloc[start_pos:(start_pos+self.batch_size)]
        # TODO: load features
        track_ids = meta_df['FileName'].values
        valid_ids, valid_features = features.compute_mfcc_example(track_ids)
        return valid_features, self.get_labels(valid_ids)

    def reset_pointer(self):
        self.pointer = 0

    def get_labels(self, name_list):
        # get labels from label dictionary
        # [[list of hihats],[list of kicks], [list of snares],
        #   [list of hihats],[list of kicks], [list of snares]],...
        print("name_list", name_list)
        labels = []
        for x in name_list:
            labels.append(self.labels[x])
        labels = np.array(labels)
        return labels


if __name__ == "__main__":
    # for test

    training_loader = dataLoader('dataset/audio_list.csv', 'dataset/labels.pkl', 32, val_set_number = 0, is_train_mode=True)
    valid_loader = dataLoader('dataset/audio_list.csv', 'dataset/labels.pkl', 32, val_set_number = 0, is_train_mode=True)


    for _ in range(training_loader.num_batch):
        track_features, label_onehot = training_loader.next_batch()
        print(len(track_features), len(label_onehot))


    for _ in range(valid_loader.num_batch):
        track_features, label_onehot = valid_loader.next_batch()






