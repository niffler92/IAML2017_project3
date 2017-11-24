import tensorflow as tf
import tensorflow.contrib.slim as slim

from tf_utils import get_activation


def layer_block(x, depth1, depth2, activation, nth_block):
    """
    Args:
        x: input
        depth1: # of filters in 1st conv layer
        depth2: # of filters in 2nd conv layer

    """
    with tf.variable_scope("layer_{}".format(nth_block)):
        conv1 = slim.conv2d(x, depth1, 2)
        conv2 = slim.conv2d(conv1, depth2, 2, activation_fn=get_activation(activation))
        conv2 = slim.max_pool2d(conv2, kernel_size=2, stride=2)

    return conv2
