
import os
import numpy as np
from scipy import stats
import pandas as pd
import librosa


def compute_mfcc_example(tids):
    successful_tids = []
    successful_features = []

    for tid in tids:
        try:
            filepath = get_audio_path('dataset/audio', tid)

            try:
                ### do not change here !
                x, sr = librosa.load(filepath, sr=44100, mono=True, duration=20)
                x = x.tolist()

                new_x = []
                while len(new_x) < 44100 * 20:
                    new_x.extend(x)
                new_x = new_x[:44100 * 20]
                x = np.array(new_x)
                ###

                stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
                del stft
                f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)


            except Exception as e:
                print(filepath, e)


            successful_tids.append(tid)
            successful_features.append(f.tolist())

        except Exception as e:
            print('{}: {}'.format(tid, repr(e)))

    return successful_tids, successful_features

def feature_examples(tid):
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

        # zero_crossing_rate
        # returns (1,t)
        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)


        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7 * 12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

        # chroma_cqt
        # returns (n_chroma, t)
        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)

        # chroma_cqt
        # returns (n_chroma, t)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
        del x

        # chroma_stft
        # returns (n_chroma, t)
        f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)

        # rmse
        # returns (1,t)
        f = librosa.feature.rmse(S=stft)

        # spectral_centroid
        # returns (1,t)
        f = librosa.feature.spectral_centroid(S=stft)

        # spectral_bandwidth
        # returns (1,t)
        f = librosa.feature.spectral_bandwidth(S=stft)

        # spectral_contrast
        # returns (n_bands+1, t)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)

        # spectral_rolloff
        # returns (1,t)
        f = librosa.feature.spectral_rolloff(S=stft)

        # mfcc
        # returns (n_mfcc, t)
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))
        return tid, 0


def get_audio_path(audio_dir, track_id):
    return os.path.join(audio_dir, track_id + '.wav')

