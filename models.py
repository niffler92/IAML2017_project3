import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
from tf_utils import get_activation
from submodule import modules


class BaseModel:
    def __init__(self, args):
        self.log = utils.get_logger("Models", None)
        self.args = args
        self.global_step = tf.Variable(0, dtype=tf.int32)

    def _create_placeholders(self):
        raise NotImplementedError

    def _create_network(self):
        raise NotImplementedError

    def _create_loss(self):
        raise NotImplementedError

    def _create_optimizer(self):
        """
        Args:
            optimizer (str): One of ["adam", "nesterov", "rmsprop", "adadelta"]
        """
        with tf.variable_scope("optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):

                if self.args['optimizer'] == "adam":
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.args['learning_rate'])
                elif self.args['optimizer'] == "nesterov":
                    optimizer = tf.train.MomentumOptimizer(learning_rate=self.args['learning_rate'],
                                                           use_nesterov=True,
                                                           momentum=self.args['momentum'])
                elif self.args['optimizer'] == "rmsprop":
                    optimizer = tf.train.RMSPropOptimizer(self.args['learning_rate'],
                                                          decay=0.9,
                                                          momentum=0.)
                elif self.args['optimizer'] == "adadelta":
                    optimizer = tf.train.AdadeltaOptimizer(self.args['learning_rate'])
                else:
                    raise ValueError("Must define more optimizers")

                def clip_if_not_none(grad):
                    if grad is None:
                        return grad
                    return tf.clip_by_value(grad, -1., 1.)
                # Gradient clipping
                gradients = optimizer.compute_gradients(self.loss_train, var_list=tf.trainable_variables())
                # clipped_grads = [(clip_if_not_none(grad), var) for grad, var in gradients]
                self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

    def _create_summaries(self):
        val_set = str(self.args['val_set_number'])

        with tf.name_scope("train"):
            summary_train = tf.summary.merge([
                tf.summary.scalar("loss"+val_set, self.loss_train),
                tf.summary.histogram("histogram_loss" + val_set, self.loss_train),
                tf.summary.scalar("accuracy"+val_set, self.acc_train)
            ])

        with tf.name_scope("valid"):
            summary_valid = tf.summary.merge([
                tf.summary.scalar("loss"+val_set, self.loss_valid),
                tf.summary.histogram("histogram_loss"+val_set, self.loss_valid),
                tf.summary.scalar("accuracy"+val_set, self.acc_valid),
            ])

        return summary_train, summary_valid

    def build_graph(self, is_training):
        """ Building graph for the model """
        self._create_placeholders()
        self._create_network(is_training=is_training)
        self._create_loss()
        self._create_optimizer()
        self.summary_train, self.summary_valid = self._create_summaries()


class CNN_BASE(BaseModel):  # FIXME Example
    def __init__(self, args):
        BaseModel.__init__(self, args)  # for python 2.7 compatibility (need to be confirmed)

    def _create_placeholders(self):
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.args['height'], self.args['width'], self.args['depth']], name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 3, 200], name="y")

    def _create_network(self, is_training):
        """is_training must be tensorflow bool type when executing train and eval at the same time.
        """
        layer1 = modules.layer_block(self.x, self.args['h4'], self.args['h4'] * self.args['h5'],
                                     self.args['activation'], 1)
        layer2 = modules.layer_block(layer1, self.args['h7'], self.args['h7'] * self.args['h8'],
                                     self.args['activation'], 2)
        layer3 = modules.layer_block(layer2, self.args['h9'], self.args['h9'] * self.args['h10'],
                                     self.args['activation'], 3)

        layer4 = slim.conv2d(layer3, 3, 1, scope="layer_4")  # , activation_fn=get_activation(self.args['activation']))
        layer4 = tf.squeeze(layer4, axis=1)
        assert layer4.shape.ndims == 3
        layer4 = tf.transpose(layer4, [0, 2, 1])  # (B, 3, 200)

        with tf.variable_scope("output"):
            self.logits = layer4
            self.y_pred = tf.cast(tf.greater(self.logits, 0), dtype=tf.float32)
            self.y_true = self.y
            correct_pred = tf.equal(self.y_pred, self.y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
            self.acc_train = accuracy
            self.acc_valid = accuracy

    def _create_loss(self):
        with tf.variable_scope("loss"):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y, logits=self.logits)
            self.loss_train = loss
            self.loss_valid = loss


class CNN(BaseModel):  # FIXME Example
    def __init__(self, args):
        BaseModel.__init__(self, args)  # for python 2.7 compatibility (need to be confirmed)

    def _create_placeholders(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.args['height'], self.args['width'], self.args['depth']], name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 3, 200], name="y")

    def _create_network(self, is_training):
        """is_training must be tensorflow bool type when executing train and eval at the same time.
        """
        layer1 = modules.layer_block(self.x, self.args['h4'], self.args['h4']*self.args['h5'], self.args['activation'], 1)
        layer2 = modules.layer_block(layer1, self.args['h7'], self.args['h7']*self.args['h8'], self.args['activation'], 2)
        layer3 = modules.layer_block(layer2, self.args['h9'], self.args['h9']*self.args['h10'], self.args['activation'], 3)

        layer4 = slim.conv2d(layer3, 3, 1, scope="layer_4")  #, activation_fn=get_activation(self.args['activation']))
        layer4 = tf.squeeze(layer4, axis=1)
        assert layer4.shape.ndims == 3
        layer4 = tf.transpose(layer4, [0, 2, 1])  # (B, 3, 200)
        layer4 = tf.nn.sigmoid(layer4)

        with tf.variable_scope("output"):
            self.logits = layer4
            self.y_pred = tf.round(self.logits)  # tf.cast(tf.greater(self.logits, 0), dtype=tf.float32)
            self.y_true = self.y
            correct_pred = tf.equal(self.y_pred, self.y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
            self.acc_train = accuracy
            self.acc_valid = accuracy

    def _create_loss(self):
        with tf.variable_scope("loss"):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y, logits=self.logits)
            self.loss_train = loss
            self.loss_valid = loss

