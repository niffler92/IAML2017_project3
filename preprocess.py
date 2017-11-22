import argparse
import numpy as np
from sklearn import preprocessing
from PIL import Image

class Preprocessor:
    """Preprocessor executes existing preprocessing related arguments.
    All preprocessing is done in DataLoader.
    """
    def __init__(self, dataset, args, is_training, feature_names):

        self.dataset = dataset
        if args is None:
            # default
            self.gamma_mel = 0.4
            self.norm_type = 2
            self.target_height = 8
        elif isinstance(args, dict):
            self.gamma_mel = args['gamma_mel']
            self.norm_type = args['norm_type']
            self.target_height = args['target_height']
        elif isinstance(args, argparse.Namespace):
            self.gamma_mel = args.gamma_mel
            self.norm_type = args.norm_type
            self.target_height = args.target_height
        else:
            raise ValueError('unknown preprocess_args')

        self.feature_names = feature_names
        self.dic_feature_len = {'mfcc': 20, 'melspectrogram': 128, 'rmse': 1}
        self.set_feature_index()


    def set_feature_index(self):
        mfcc_order = self.feature_names.index('mfcc')
        melspectrogram_order = self.feature_names.index('melspectrogram')
        rmse_order = self.feature_names.index('rmse')

        features_size = [self.dic_feature_len[f] for f in self.feature_names]

        feature_index_list = []
        offset = 0
        for feature_size in features_size:
            feature_index_list.append(range(offset, offset+feature_size))
            offset += feature_size

        self.mfcc_index = feature_index_list[mfcc_order]
        self.mel_index = feature_index_list[melspectrogram_order]
        self.rmse_index = feature_index_list[rmse_order]


    @staticmethod
    def mel_spectrogram_scale(X, gamma_mel, mel_index):
        """
        gamma correction of mel spectrogram
        gamma_mel : (0,1] float
        """
        if gamma_mel == 1.0:
            return X
        dataset_mel = X[:, mel_index, :]
        X[:, mel_index, :] = np.power(dataset_mel, gamma_mel)
        return X


    @staticmethod
    def normalization(X, norm_type, feature_indices):
        """
        Normalization
        norm_type 0: no Normalization
        norm_type 1: Normalization for each feature
        norm_type 2: Normalization for each y line
        norm_type 3: mfcc(for feature), others(for each y line)
        """

        mfcc_index = feature_indices['mfcc']
        mel_index = feature_indices['mel']
        rmse_index = feature_indices['rmse']

        if norm_type == 0:
            return
        elif norm_type == 1:
            features_mfcc = X[:, mfcc_index, :]
            features_mel = X[:, mel_index, :]
            features_rmse = X[:, rmse_index, :]
            mfcc_avg = np.average(features_mfcc)
            mel_avg = np.average(features_mel)
            rmse_avg = np.average(features_rmse)
            mfcc_std = np.std(features_mfcc)
            mel_std = np.std(features_mel)
            rmse_std = np.std(features_rmse)
            X[:, mfcc_index, :] = (features_mfcc - mfcc_avg) / mfcc_std
            X[:, mel_index, :] = (features_mel - mel_avg) / mel_std
            X[:, rmse_index, :] = (features_rmse - rmse_avg) / rmse_std
        elif norm_type == 2:
            feature_size = np.shape(X)[1]
            features = np.transpose(X, axes=[1, 0, 2])
            features = np.reshape(features, [feature_size, -1])
            features_avg = np.average(features, axis=1)
            features_std = np.std(features, axis=1)
            features_avg = np.expand_dims(features_avg, 1)
            features_avg = np.expand_dims(features_avg, 0)
            features_std = np.expand_dims(features_std, 1)
            features_std = np.expand_dims(features_std, 0)
            X = (X - features_avg) / features_std
        elif norm_type == 3:
            features_mfcc = X[:, mfcc_index, :]
            mfcc_avg = np.average(features_mfcc)
            mfcc_std = np.std(features_mfcc)
            X[:, mfcc_index, :] = (features_mfcc - mfcc_avg) / mfcc_std

            feature_index = np.concatenate([mel_index, rmse_index])
            features = np.transpose(X[:,feature_index,:], axes=[1, 0, 2])
            feature_size = np.shape(feature_index)[0]
            features = np.reshape(features, [feature_size, -1])
            features_avg = np.average(features, axis=1)
            features_std = np.std(features, axis=1)
            features_avg = np.expand_dims(features_avg, 1)
            features_avg = np.expand_dims(features_avg, 0)
            features_std = np.expand_dims(features_std, 1)
            features_std = np.expand_dims(features_std, 0)
            X[:,feature_index,:] = (X[:,feature_index,:] - features_avg) / features_std
        else:
            raise ValueError('norm_type(%s) not defined'%norm_type)

        return X


    @staticmethod
    def resize_time_length(X):
        new_time_len = 1600
        data_shape = np.shape(X)
        time_len = data_shape[2]
        dataset_reshape = np.transpose(X, [2, 0, 1])
        dataset_reshape = np.reshape(dataset_reshape, [time_len, -1])
        img = Image.fromarray(dataset_reshape)
        img = img.resize([img.width, new_time_len], Image.BILINEAR)
        dataset_new = np.asarray(img)
        dataset_reshape = np.reshape(dataset_new, [new_time_len, data_shape[0], data_shape[1]])
        X = np.transpose(dataset_reshape, [1, 2, 0])
        return X




    @staticmethod
    def height_to_channel(X, target_height, feature_indices):

        def _height_resize(data, h_new):
            if np.ndim(data) == 2:
                data = np.expand_dims(data, axis=1)

            h_old = np.shape(data)[0]
            ratio = float(h_new) / h_old
            if ratio == 1.0:
                return data
            elif ratio < 1.0:
                resample_type = Image.BOX
            else:
                resample_type = Image.BILINEAR

            shape_org = np.shape(data)
            data = np.transpose(data, [1, 0, 2])
            data = np.reshape(data, [np.shape(data)[0], -1])

            img = Image.fromarray(data)
            img = img.resize([img.width, h_new], resample_type)
            data_new = np.asarray(img)
            data_new = np.reshape(data_new, [h_new, shape_org[0], shape_org[2]])
            data_new = np.transpose(data_new, [1, 0, 2])
            return data_new

        mfcc_index = feature_indices['mfcc']
        mel_index = feature_indices['mel']
        rmse_index = feature_indices['rmse']
        features_mfcc = X[:, mfcc_index, :]
        features_mel = X[:, mel_index, :]
        features_rmse = X[:, rmse_index, :]

        # features_mfcc1 = height_resize(features_mfcc[:, 0:10, :], 30)
        # features_mfcc2 = height_resize(features_mfcc[:, 10:18, :], 16)
        # features_mfcc3 = height_resize(features_mfcc[:, 18:20, :], 2)
        # features_mfcc = np.concatenate([features_mfcc1, features_mfcc2, features_mfcc3], axis=1)
        features_mfcc1 = _height_resize(features_mfcc[:, 0:16, :], 16)
        features_mfcc2 = _height_resize(features_mfcc[:, 16:20, :], 8)
        features_mfcc = np.concatenate([features_mfcc1, features_mfcc2], axis=1)

        features_mel1 = _height_resize(features_mel[:, 0:8, :], 16)
        features_mel2 = _height_resize(features_mel[:, 16:24, :], 16)
        features_mel3 = _height_resize(features_mel[:, 24:88, :], 32)
        features_mel4 = _height_resize(features_mel[:, 88:128, :], 8)
        features_mel = np.concatenate([features_mel1, features_mel2, features_mel3, features_mel4], axis=1)

        feagures_rmse = _height_resize(features_rmse[:, 0, :], 8)

        features_mfcc = np.transpose(features_mfcc, axes=[0, 2, 1])
        features_mel = np.transpose(features_mel, axes=[0, 2, 1])
        feagures_rmse = np.transpose(feagures_rmse, axes=[0, 2, 1])

        len0 = np.shape(features_mfcc)[0]
        len1 = np.shape(features_mfcc)[1]
        features_mfcc = np.reshape(features_mfcc, [len0, len1, -1, target_height])
        features_mel = np.reshape(features_mel, [len0, len1, -1, target_height])
        feagures_rmse = np.reshape(feagures_rmse, [len0, len1, -1, target_height])

        X = np.concatenate([features_mfcc, features_mel, feagures_rmse], axis=2)
        X = np.transpose(X, [0, 3, 1, 2])
        return X


    def run(self):
        """Normalizing sequence may be important
        """
        feature_indices = {'mfcc':self.mfcc_index, 'mel':self.mel_index, 'rmse':self.rmse_index}
        X = self.dataset

        X = Preprocessor.mel_spectrogram_scale(X, self.gamma_mel, self.mel_index)
        X = Preprocessor.normalization(X, self.norm_type, feature_indices)
        X = Preprocessor.resize_time_length(X)
        X = Preprocessor.height_to_channel(X, self.target_height, feature_indices)

        return X
