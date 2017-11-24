import argparse
import numpy as np
import matplotlib.pyplot as plt

class Augmentation:
    """Augmentation executes existing data augmentation related arguments.
    All augmentation is done in DataLoader.
    """
    def __init__(self, features, args, is_training):
        self.features = features
        self.is_training = is_training
        if args is None:
            # default
            self.max_noise = 0.4
        elif isinstance(args, dict):
            self.max_noise = args['max_noise']
        elif isinstance(args, argparse.Namespace):
            self.max_noise = args.max_noise
        else:
            raise ValueError('unknown augmentation_args')


    @staticmethod
    def add_gaussian_noise(X, max_noise, is_training):
        if is_training == False:
            return X

        data_std = np.std(X)
        noise_std = np.random.uniform() * data_std * max_noise
        X += np.random.normal(scale = noise_std, size=np.shape(X))
        return X


    def run(self):

        X = self.features
        X = Augmentation.add_gaussian_noise(X, self.max_noise, self.is_training)

        return X
