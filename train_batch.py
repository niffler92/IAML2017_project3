
from dataloader import DataLoader
import tensorflow as tf
from params import param_batch_cnn100 as param
import utils
from train import *



def train_batch():
  param_list_dict = param.param_list_dict()
  param_dict = utils.get_random_param(param_list_dict)

  train_dataloader = DataLoader(is_training=True)
  valid_dataloader = DataLoader(is_training=False)

  while True:
    param_dict = utils.get_random_param(param_list_dict)

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    valid_f1_list = []
    epoch_list = []

    session = tf.Session()

    # for val_set_number in range(3):
    for val_set_number in range(1):
      # param_dict['val_set_number'] = val_set_number
      param_dict['val_set_number'] = np.random.randint(3)

      model = utils.find_class_by_name([models], param_dict['model'])(param_dict)
      model.build_graph(is_training=tf.constant(True, dtype=tf.bool))

      train_dataloader.reset_args(param_dict)
      valid_dataloader.reset_args(param_dict)
      print(param_dict)

      train_loss, train_acc, valid_loss, valid_acc, valid_f1, epoch = train(model, session, train_dataloader, valid_dataloader, param_dict)
      train_loss_list.append(train_loss)
      train_acc_list.append(train_acc)
      valid_loss_list.append(valid_loss)
      valid_acc_list.append(valid_acc)
      valid_f1_list.append(valid_f1)
      epoch_list.append(epoch)

    train_loss = np.average(train_loss_list)
    train_acc = np.average(train_acc_list)
    valid_loss = np.average(valid_loss_list)
    valid_acc = np.average(valid_acc_list)
    valid_f1 = np.average(valid_f1_list)
    epoch = np.max(epoch_list)

    session.close()

    tf.reset_default_graph()

    print('train_loss(%.4f), train_acc(%.4f), valid_loss(%.4f), valid_acc(%.4f), valid_f1(%.4f), epoch(%d)'
          % (train_loss, train_acc, valid_loss, valid_acc, valid_f1, epoch))
    utils.write_param(param_dict, train_loss, train_acc, valid_loss, valid_acc, valid_f1, epoch)

if __name__ == "__main__":
  train_batch()
