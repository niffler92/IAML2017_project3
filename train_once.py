
from dataloader import DataLoader
import tensorflow as tf
from params import param_cnn100 as param
import utils
from train import *



def train_batch(tag_label):
  param_list_dict = param.param_list_dict()
  param_dict = utils.get_random_param(param_list_dict)

  train_dataloader = DataLoader(is_training=True)
  valid_dataloader = DataLoader(is_training=False)

  model = utils.find_class_by_name([models], param_dict['model'])(param_dict)
  model.build_graph(is_training=tf.constant(True, dtype=tf.bool))

  param_dict = utils.get_random_param(param_list_dict)

  # tag_label = 'CNN100_try1' # default
  # tag_label = 'CNN100_try2' # bn_layers = 7 -> 0 # local min
  # tag_label = 'CNN100_try3' # loss_reduce_max_index = 1 -> 0 # local min
  # tag_label = 'CNN100_try4' # optimizer = adam -> rmsprop # nan
  # tag_label = 'CNN100_try5' # bn_layers = 0 -> 7, loss_reduce_max_index = 0 -> 1 # local min
  # tag_label = 'CNN100_try6' # momentum = 0.99 -> 0.9
  # tag_label = 'CNN100_try7' # momentum = 0.9 -> 0.99, bn_layers = 7 -> 0, loss_reduce_max_index = 1 -> 0
  # tag_label = 'CNN100_try8' # bn_layers = 0 -> 7, loss_reduce_max_index = 0 -> 1
  # tag_label = 'CNN100_try9' # loss 0/1 weight



  train_loss_list = []
  train_acc_list = []
  valid_loss_list = []
  valid_acc_list = []
  valid_f1_list = []
  epoch_list = []

  session = tf.Session()

  for val_set_number in range(3):
    param_dict['val_set_number'] = val_set_number
    param_dict['tag_label'] = tag_label + str(val_set_number)
    train_dataloader.reset_args(param_dict)
    valid_dataloader.reset_args(param_dict)
    print(param_dict)

    train_loss, train_acc, valid_loss, valid_acc, f1_best, epoch = train(model, session, train_dataloader, valid_dataloader, param_dict)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_acc)
    valid_f1_list.append(f1_best)
    epoch_list.append(epoch)

  train_loss = np.average(train_loss_list)
  train_acc = np.average(train_acc_list)
  valid_loss = np.average(valid_loss_list)
  valid_acc = np.average(valid_acc_list)
  valid_f1 = np.average(valid_f1_list)
  epoch = np.max(epoch_list)

  session.close()

  print('train_loss(%.4f), train_acc(%.4f), valid_loss(%.4f), valid_acc(%.4f), valid_f1(%.4f),epoch(%d)'
        % (train_loss, train_acc, valid_loss, valid_acc, valid_f1, epoch))

if __name__ == "__main__":

  start_index = 5000
  while True:
    tag_label = 'CNN100_try' + str(start_index)
    train_batch(tag_label)
    start_index += 1
    tf.reset_default_graph()
