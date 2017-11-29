import numpy as np


def range_float(min, max, step):
  step_num = int((max - min) / step + 0.000001)
  return (np.asarray(range(step_num + 1)) / float(step_num) * (max - min) + min).tolist()


def param_list_dict():
  param_list_dict = dict()

  # fixed
  param_list_dict['model'] = ['CNN100']
  param_list_dict['feature_names'] = [["mfcc", "melspectrogram", "rmse"]]
  param_list_dict['height'] = [8]
  param_list_dict['width'] = [1600]
  param_list_dict['depth'] = [13]

  # preprocessing - fixed
  param_list_dict['gamma_mel'] = [0.3]
  param_list_dict['norm_type'] = [2]  # 0(no norm), 1(feature-wise), 2(line-wise), 3(mfcc(feature-wise), others(line-wise))
  param_list_dict['target_height'] = [8]

  # optim 1
  # param_list_dict['batch_size'] = [1, 2, 4, 8, 16]  # [8, 16, 32, 64]
  # param_list_dict['activation'] = ['relu', 'relu6', 'swish']
  # param_list_dict['l2_loss_scale'] = [0.01, 0.1, 1.0]
  # param_list_dict['loss_reduce_max_index'] = [0, 1, 2]  # 0(total mean), 1(max 1), 2(max 2)
  # param_list_dict['focal_loss_gamma_list'] = [0, 0.5, 1, 2, 5]  # (0 == cross entropy)  #  [0, 0.5, 1, 2, 5]
  # param_list_dict['bn_layers'] = [0, 1, 2, 3, 4, 5, 6, 7]  # 0~8
  # param_list_dict['loss1_weight'] = range_float(0.5, 0.95, 0.05)
  param_list_dict['batch_size'] = [8]  # [8, 16, 32, 64]
  param_list_dict['activation'] = ['relu6']
  param_list_dict['l2_loss_scale'] = [0.1]
  param_list_dict['loss_reduce_max_index'] = [0]  # 0(total mean), 1(max 1), 2(max 2)
  param_list_dict['focal_loss_gamma_list'] = [1.0]  # (0 == cross entropy)  #  [0, 0.5, 1, 2, 5]
  param_list_dict['bn_layers'] = [4]  # 0~8
  param_list_dict['loss1_weight'] = [0.8]

  # optim 2
  # param_list_dict['h4'] = [16, 32, 64, 128]
  # param_list_dict['extra_1x1_conv'] = [32, 64, 128]  # [64]
  param_list_dict['h4'] = [16]
  param_list_dict['extra_1x1_conv'] = [128]  # [64]=





  # Augmentation - no optim
  # param_list_dict['max_noise'] = [0.5]
  # param_list_dict['optimizer'] = ['rmsprop']  # ['adam', 'nesterov', 'rmsprop', 'adadelta']
  # param_list_dict['dropout'] = [0.5]
  # param_list_dict['learning_rate'] = [1e-2] # [1e-2, 1e-3, 1e-4]
  # param_list_dict['momentum'] = [0.99]
  param_list_dict['max_noise'] = range_float(0, 0.7, 0.1)
  param_list_dict['optimizer'] = ['adam', 'nesterov', 'rmsprop', 'adadelta']
  param_list_dict['dropout'] = range_float(0.5, 0.8, 0.1)
  param_list_dict['learning_rate'] = [1e-2, 1e-3] # [1e-2, 1e-3, 1e-4]
  param_list_dict['momentum'] = [0.95, 0.97, 0.99]


  # configs
  param_list_dict['val_set_number'] = [0]
  param_list_dict['checkpoint_path'] = ['']
  param_list_dict['train_dir'] = ['train_dir']
  param_list_dict['tag_label'] = ['default']
  param_list_dict['step_save_summaries'] = [100000000]  # no save
  param_list_dict['max_epochs'] = [100]  # 300 / 1000
  param_list_dict['no_save_ckpt'] = [True]
  param_list_dict['train_batch_result_filename'] = ['./log/batch_log/result.txt']
  param_list_dict['valid_epoch'] = [10]

  return param_list_dict
