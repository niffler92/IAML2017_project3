
from dataloader import DataLoader
import tensorflow as tf
from params import param_batch as param
import utils
from train import *


def train_batch():
  param_list_dict = param.param_list_dict()
  param_dict = utils.get_random_param(param_list_dict)

  train_dataloader = DataLoader(is_training=True)
  valid_dataloader = DataLoader(is_training=False)

  model = utils.find_class_by_name([models], param_dict['model'])(param_dict)
  model.build_graph(is_training=tf.constant(True, dtype=tf.bool))

  while True:
    param_dict = utils.get_random_param(param_list_dict)

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    epoch_list = []

    for val_set_num in range(3):
      param_dict['val_set_number'] = val_set_num
      train_dataloader.reset_args(param_dict)
      valid_dataloader.reset_args(param_dict)
      print(param_dict)

      train_loss, train_acc, valid_loss, valid_acc, epoch = train(model, train_dataloader, valid_dataloader, param_dict)
      train_loss_list.append(train_loss)
      train_acc_list.append(train_acc)
      valid_loss_list.append(valid_loss)
      valid_acc_list.append(valid_acc)
      epoch_list.append(epoch)

    train_loss = np.average(train_loss_list)
    train_acc = np.average(train_acc_list)
    valid_loss = np.average(valid_loss_list)
    valid_acc = np.average(valid_acc_list)
    epoch = np.max(epoch_list)

    print('train_loss(%.4f), train_acc(%.4f), valid_loss(%.4f), valid_acc(%.4f), epoch(%d)' % (train_loss, train_acc, valid_loss, valid_acc, epoch))
    utils.write_param(param_dict, train_loss, train_acc, valid_loss, valid_acc, epoch)

if __name__ == "__main__":
  train_batch()
