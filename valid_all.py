
from dataloader import DataLoader
import tensorflow as tf
from params import param_base as param
import utils
from train import *



def train_batch():
  param_list_dict = param.param_list_dict()

  valid_dataloader = DataLoader(is_training=False, total_validation=True)
  param_dict = utils.get_random_param(param_list_dict)
  valid_dataloader.reset_args(param_dict)

  model = utils.find_class_by_name([models], param_dict['model'])(param_dict)
  model.build_graph(is_training=tf.constant(True, dtype=tf.bool))

  max_f1_score = 0
  max_acc = 0

  for i in range(3):
    checkpoint_path = os.path.join('checkpoint', 'CNN_BASE', 'CNN-bs32-val' + str(i))
    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore(session, checkpoint_path)
    avg_acc, avg_loss, avg_f1_score = valid_full(0, model, valid_dataloader, session, param_dict)
    session.close()

    if max_f1_score < avg_f1_score:
      max_f1_score = avg_f1_score
      max_acc = avg_acc
      best_ckpt_name = checkpoint_path

  print('best_ckpt_name(%s), avg_acc(%.4f), avg_f1_score(%.4f)' % (best_ckpt_name, max_acc, max_f1_score))

if __name__ == "__main__":
  train_batch()
