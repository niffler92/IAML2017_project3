import os
import time

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed
import tensorflow as tf
import matplotlib.pyplot as plt

from settings import PROJECT_ROOT


def load_features(feature_name, tids):

    successful_tids = []
    successful_features = []
    for tid in tids:
        try:
            pkl_path = get_audio_path(
                'dataset/audio', tid).replace('.mp3', '_{}.pkl'.format(feature_name))
            feature = np.load(pkl_path)
            successful_features.append(feature)
            successful_tids.append(tid)

        except Exception as e:
            print('{}: {}'.format(tid, repr(e)))
            return 0, tid

    successful_features = np.array(successful_features)
    successful_tids = np.array(successful_tids)
    return successful_features, successful_tids


def compute_feature(feature_name, tid):
    # example of various librosa features
    # please check [https://librosa.github.io/librosa/feature.html]
    threshold = 1278900
    try:
        filepath = get_audio_path('dataset/audio', tid)
        ### do not change here !
        x, sr = librosa.load(filepath, sr=44100, mono=True, duration=20)
        x = x.tolist()
        origin_length = len(x)

        new_x = []
        while len(new_x) < 44100 * 20:
            new_x.extend(x)
        new_x = new_x[:44100 * 20]
        x = np.array(new_x)
        ###

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7 * 12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
        # zero_crossing_rate
        # returns (1,t)
        if feature_name == 'zero_crossing_rate':
            f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        # chroma_cqt
        # returns (n_chroma, t)
        elif feature_name == 'chroma_cqt':
            f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        # chroma_cens
        # returns (n_chroma, t)
        elif feature_name == 'chroma_cens':
            f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        # chroma_stft
        # returns (n_chroma, t)
        elif feature_name == 'chroma_stft':
            f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
        # rmse
        # returns (1,t)
        elif feature_name == 'rmse':
            f = librosa.feature.rmse(S=stft)
        # spectral_centroid
        # returns (1,t)
        elif feature_name == 'spectral_centroid':
            f = librosa.feature.spectral_centroid(S=stft)
        # spectral_bandwidth
        # returns (1,t)
        elif feature_name == 'spectral_bandwidth':
            f = librosa.feature.spectral_bandwidth(S=stft)
        # spectral_contrast
        # returns (n_bands+1, t)
        elif feature_name == 'spectral_contrast':
            f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        # spectral_rolloff
        # returns (1,t)
        elif feature_name == 'spectral_rolloff':
            f = librosa.feature.spectral_rolloff(S=stft)
        # melspectrogram
        # returns not checked
        elif feature_name == 'melspectrogram':
            f = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
            # f = librosa.power_to_db(f, ref=np.max)
        # mfcc
        # returns (n_mfcc, t)
        elif feature_name == 'mfcc':
            mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
            f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
            del mel

        del x
        del cqt
        del stft

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))
        return None, tid

    return f, tid


def get_audio_path(audio_dir, track_id):
    return os.path.join(PROJECT_ROOT, audio_dir, track_id + '.wav')


def maybe_create_data_file(feature_name, all_tids):
    data_file_path = os.path.join(PROJECT_ROOT, 'dataset/{}.pkl'.format(feature_name))
    if not tf.gfile.Exists(data_file_path):
        st = time.time()
        print("{}.pkl not found in dataset directory. "
              "Creating pickle file...".format(feature_name))

        feature_list = []
        try:
            for tid in tqdm(all_tids):
                feature, _ = compute_feature(feature_name, tid)
                if feature is None:
                    raise ValueError('tid %s has feature error')
                else:
                    feature_list.append(feature)

            print("Computing features took {} seconds.".format(time.time() - st))
            feature_list = np.array(feature_list)

            print("Dumping {}.pkl with array size {}".format(feature_name, feature_list.size))
            feature_list.dump(data_file_path)
        except Exception as e:
            print("track_id {}: {}".format(tid, repr(e)))
    else:
        print("Feature data file exists. Use existing ones..")


def create_all_features(is_parallel=True, is_individual=False, n_jobs=8):
    feature_names = ['chroma_cens', 'chroma_stft', 'chroma_cqt', 'rmse',
                     'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast',
                     'melspectrogram', 'mfcc', 'zero_crossing_rate']
    # metadata
    all_tids = pd.read_csv('dataset/audio_list.csv').FileName.tolist()
    # TODO: Labels should be ordered with the order of audio_list.csv in dataloader

    print("Creating features")
    if is_parallel:
        Parallel(n_jobs=n_jobs)(
            delayed(maybe_create_data_file)(feature, all_tids) for feature in feature_names)
    else:
        for feature_name in feature_names:
            maybe_create_data_file(feature_name, all_tids)

    print("Features were successfully created.")


if __name__ == '__main__':
    create_all_features(is_parallel=True)
