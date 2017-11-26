import numpy as np


def range_float(min, max, step):
  return (np.asarray(range(step+1)) / float(step) * (max-min) + min).tolist()


def param_list_dict():

  param_list_dict = dict()

  # model
  param_list_dict['model'] = ['CNN']
  param_list_dict['feature_names'] = [["mfcc", "melspectrogram", "rmse"]]

  # preprocessing
  param_list_dict['gamma_mel'] = range_float(0, 1, 0.01)
  param_list_dict['norm_type'] = [0, 1, 2, 3]  # 0(no norm), 1(feature-wise), 2(line-wise), 3(mfcc(feature-wise), others(line-wise))
  param_list_dict['target_height'] = [8]

  # Augmentation
  param_list_dict['max_noise'] = range_float(0, 1, 0.01)

  # model

  # hyperparameters
  param_list_dict['batch_size'] = [8, 16, 32, 64]                 # [8, 16, 32, 64]
  param_list_dict['h4'] = [8, 16, 32, 64]
  param_list_dict['h5'] = [0.5, 1, 2]
  param_list_dict['h7'] = [8, 16, 32, 64]
  param_list_dict['h8'] = [0.5, 1, 2]
  param_list_dict['h9'] = [8, 16, 32, 64]
  param_list_dict['h10'] = [0.5, 1, 2]
  param_list_dict['height'] = [8]
  param_list_dict['width'] = [1600]
  param_list_dict['depth'] = [13]
  param_list_dict['activation'] = ['relu', 'relu6', 'swish']

  # train
  param_list_dict['learning_rate'] = [1e-2, 1e-3, 1e-4]
  param_list_dict['optimizer'] = ['adam']
  param_list_dict['momentum'] = [0.9, 0.95, 0.97, 0.99]
  param_list_dict['dropout'] = range_float(0.2, 0.8, 0.1)

  # configs
  param_list_dict['val_set_number'] = [0]
  param_list_dict['checkpoint_path'] = ['']
  param_list_dict['train_dir'] = ['train_dir']
  param_list_dict['tag_label'] = ['default']
  param_list_dict['step_save_summaries'] = [100000000]  # no save
  param_list_dict['max_epochs'] = [500]
  param_list_dict['no_save_ckpt'] = [True]
  param_list_dict['train_batch_result_filename'] = ['./log/batch_log/result.txt']
  param_list_dict['valid_epoch'] = [10]



  return param_list_dict
