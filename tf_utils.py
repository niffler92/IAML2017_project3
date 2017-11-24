import tensorflow as tf


def get_activation(name):
    if name == "relu":
        return tf.nn.relu
    elif name == "relu6":
        return tf.nn.relu6
    elif name == "swish":
        pass
