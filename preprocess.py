import numpy as np
from sklearn import preprocessing


class Preprocessor:
    """Preprocessor executes existing preprocessing related arguments.
    All preprocessing is done in DataLoader.
    """
    @staticmethod
    def normalize(X, norm_type='l2'):
        """
        axis=1: independently normalize each sample
        axis=0: normalize each feature
        """
        shape = X.shape  # (B, H, W, C)
        X_reshaped = X.reshape([-1, np.prod(shape[1:])])  # (B, H*W)
        X_normalized = preprocessing.normalize(X=X_reshaped, axis=1, norm=norm_type)
        return X_normalized.reshape(shape)

    @staticmethod
    def normalize_by_row(X, norm_type='l2'):
        """Our plan
        """
        shape = X.shape
        if shape[-1] == 1:
            X = np.squeeze(X)  # (B, H, W)
        X_reshaped = X.reshape([shape[1], -1])  # (H, B*W)
        # normalize along H
        X_normalized = preprocessing.normalize(X=X_reshaped, axis=1, norm=norm_type)
        return X_normalized.reshape(shape)

    @staticmethod
    def bilinear_resize(X):
        return X

    @staticmethod
    def split_to_blocks(X, y, blocks=4):
        return X, y

    def run(self, X, args, is_training):
        """Normalizing sequence may be important
        """
        if isinstance(args, argparse.Namespace):
            if args.normalize == True:
                assert args.norm_type is not None
                X = self.normalize(X, norm_type)
            if args.normalize_by_row == True:
                assert args.norm_type is not None
                X = self.normalize_by_row(X, norm_type)
            if args.bilinear == True:
                X = self.bilinear_resize(X)

        elif isinstance(args, dict):
            pass

        return self.X
