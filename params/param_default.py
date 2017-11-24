import numpy as np

def range_float(min, max, step):
  return (np.asarray(range(step+1)) / float(step) * (max-min) + min).tolist()


def param_list_dict():

  param_list_dict = dict()

  # model
  param_list_dict['model'] = ['model']
  param_list_dict['feature_names'] = [["mfcc", "melspectrogram", "rmse"]]

  # preprocessing
  param_list_dict['gamma_mel'] = [0.4]
  param_list_dict['norm_type'] = [2]  # 0(no norm), 1(feature-wise), 2(line-wise), 3(mfcc(feature-wise), others(line-wise))
  param_list_dict['target_height'] = [8]

  # Augmentation
  param_list_dict['max_noise'] = [1.0]

  # model
  param_list_dict['input_time_len_list'] = [1024]

  # hyperparameters
  param_list_dict['batch_size'] = [32]                 # [8, 16, 32, 64]

  # train
  param_list_dict['learning_rate'] = [1e-3]
  param_list_dict['optimizer'] = ['adam']
  param_list_dict['momentum'] = [0.99]
  param_list_dict['dropout'] = [0.5]

  # configs
  param_list_dict['val_set_number'] = [0]
  param_list_dict['checkpoint_path'] = ['']
  param_list_dict['train_dir'] = ['train_dir/default']
  param_list_dict['step_save_summaries'] = [100000000]  # no save
  param_list_dict['max_epochs'] = [500]
  param_list_dict['no_save_ckpt'] = [True]

  return param_list_dict

