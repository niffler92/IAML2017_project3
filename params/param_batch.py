import numpy as np


def range_float(min, max, step):
  return (np.asarray(range(step+1)) / float(step) * (max-min) + min).tolist()


def param_list_dict():

  param_list_dict = dict()

  # model
  param_list_dict['model'] = ['CNN', 'CNN100']
  param_list_dict['feature_names'] = [["mfcc", "melspectrogram", "rmse"]]

  # preprocessing
  param_list_dict['gamma_mel'] = range_float(0, 1, 100)
  param_list_dict['norm_type'] = [0, 1, 2, 3]  # 0(no norm), 1(feature-wise), 2(line-wise), 3(mfcc(feature-wise), others(line-wise))
  param_list_dict['target_height'] = [8]

  # Augmentation
  param_list_dict['max_noise'] = range_float(0, 1, 100)

  # model
  param_list_dict['input_time_len_list'] = [1024]

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
  param_list_dict['activation'] = ['relu6', 'swish']

  # CNN100 ( Doesn't matter to stay together )
  param_list_dict['l2_loss_scale'] = [0.1]
  param_list_dict['loss_reduce_max_index'] = [0]  # 0(total mean), 1(max 1), 2(max 2)
  param_list_dict['focal_loss_gamma_list'] = [1.0]  # (0 == cross entropy)  #  [0, 0.5, 1, 2, 5]
  param_list_dict['bn_layers'] = [4]  # 0~8
  param_list_dict['loss1_weight'] = [0.8]
  param_list_dict['extra_1x1_conv'] = [128]  # [64]

  # train
  param_list_dict['learning_rate'] = [1e-2, 1e-3, 1e-4]
  param_list_dict['optimizer'] = ['adam', 'rmsprop']
  param_list_dict['momentum'] = [0.9, 0.95, 0.97, 0.99]
  param_list_dict['dropout'] = range_float(0.2, 0.8, 6)

  # configs
  param_list_dict['val_set_number'] = [0]
  param_list_dict['checkpoint_path'] = ['']
  param_list_dict['train_dir'] = ['train_dir']
  param_list_dict['tag_label'] = ['experiment']
  param_list_dict['step_save_summaries'] = [10]  # no save
  param_list_dict['max_epochs'] = [1500]
  param_list_dict['no_save_ckpt'] = [False]
  param_list_dict['train_batch_result_filename'] = ['./log/batch_log/result_experiment.txt']



  return param_list_dict
