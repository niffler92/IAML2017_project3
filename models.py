import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils

class BaseModel:
    def __init__(self,args):
        self.log = utils.get_loger("Models", None)
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

                if self.args.optimizer == "adam":
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
                elif self.args.optimizer == "nesterov":
                    optimizer = tf.train.MomentumOptimizer(learning_rate=self.args.learning_rate,
                                                           use_nesterov=True,
                                                           momentum=self.args.momentum)
                elif self.args.optimizer == "rmsprop":
                    optimizer = tf.train.RMSPropOptimizer(self.args.learning_rate,
                                                          decay=0.9,
                                                          momentum=0.)
                elif self.args.optimizer == "adadelta":
                    optimizer = tf.train.AdadeltaOptimizer(self.args.learning_rate)
                else:
                    raise ValueError("Must define more optimizers")

                def clip_if_not_none(grad):
                    if grad is None:
                        return grad
                    return tf.clip_by_value(grad, -1., 1.)
                # Gradient clipping
                gradients = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables())
                # clipped_grads = [(clip_if_not_none(grad), var) for grad, var in gradients]
                self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

    def build_graph(self, is_training):
        """ Building graph for the model """
        self._create_placeholders()
        self._create_network(is_training=is_training)
        self._create_loss()
        self._create_optimizer()


class ConvPool(BaseModel):  # FIXME Example
    def __init__(self, args):
        super().__init__(args)

    def _create_placeholders(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.args.height, self.args.width, 1], name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.args.num_classes], name="y")

    def _create_network(self, is_training, regularization_scale=None):
        """is_training must be tensorflow bool type when executing train and eval at the same time.
        """
        if regularization_scale:
            regularizer = tf.contrib.layers.l2_regularizer(scale=regularization_scale)
        else:
            regularizer = None

        with tf.variable_scope('conv1'):
            conv1_h = self.x.shape.as_list()[1]
            conv1 = tf.layers.conv2d(self.x, filters=256,
                                     kernel_size=[conv1_h, 4],
                                     strides=[1, 1],
                                     kernel_regularizer=regularizer,
                                     padding="valid")
            conv1 = tf.layers.batch_normalization(conv1,
                                                  training=is_training,
                                                  trainable=True)
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.transpose(conv1, [0, 3, 2, 1])
            conv1 = tf.layers.max_pooling2d(conv1,
                                            pool_size=[1, 4],
                                            strides=[1, 4],
                                            padding="valid")

        with tf.variable_scope("temp_pool"):
            w_conv3 = conv3.shape.as_list()[2]
            conv3_avg = tf.nn.pool(conv3, pooling_type="AVG",
                                   window_shape=[1, w_conv3],
                                   strides=[1, 1],
                                   padding="VALID",
                                   name="avg_pool")
            conv3_max = tf.layers.max_pooling2d(conv3,
                                                pool_size=[1, w_conv3],
                                                strides=[1, 1],
                                                padding="VALID")
            conv3_l2 = tf.norm(conv3, axis=2, keep_dims=True)
            concat = tf.concat([conv3_avg, conv3_max, conv3_l2], 3)

        with tf.variable_scope("fc1"):
            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(concat)
            fc1 = tf.layers.dense(fc1, 1024, # 2048
                                  kernel_regularizer=regularizer)
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.layers.batch_normalization(fc1, training=is_training, trainable=True)
            fc1 = tf.layers.dropout(fc1, rate=self.args.dropout, training=is_training)

        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(fc2, self.args.num_classes, name="logits")
            self.y_pred = tf.argmax(self.logits, 1, name="y_pred")
            self.y_true = tf.argmax(self.y, 1, name="y_true")
            correct_pred = tf.equal(self.y_pred, self.y_true)
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")

    def _create_loss(self):
        with tf.variable_scope("loss"):
            self.loss = tf.losses.softmax_cross_entropy(self.y, logits=self.logits)
