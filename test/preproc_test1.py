
from dataloader import DataLoader
import tensorflow as tf
from params import param_cnn100 as param
import utils
from train import *
from params.param_cnn100 import range_float
from preprocess import Preprocessor
import matplotlib.pyplot as plt
import copy
import librosa
import numpy as np

param_list_dict = param.param_list_dict()
param_dict = utils.get_random_param(param_list_dict)

dataloader = DataLoader(is_training=True)

model = utils.find_class_by_name([models], param_dict['model'])(param_dict)
model.build_graph(is_training=tf.constant(True, dtype=tf.bool))

param_dict = utils.get_random_param(param_list_dict)


dataloader.args = param_dict
dataloader.batch_size = 1
dataloader.val_set_number = utils.get_arg(param_dict, 'val_set_number')
dataloader.feature_names = utils.get_arg(param_dict, 'feature_names')
dataloader.create_dataset_org()

dataset = copy.deepcopy(dataloader.dataset_org)
dataloader.metadata_df = dataloader.metadata_df_org.copy(deep=True)
dataloader.create_batches()  # Get Train/Validation from original

dataloader.args['norm_type'] = 2

img_index = 3
for gamma in range_float(0.1, 0.9, 0.1):
  dataloader.args['gamma_mel'] = gamma
  dataloader.args['gamma_mel'] = 0.2

  dataset = copy.deepcopy(dataloader.dataset_org)
  preprocessor = Preprocessor(dataset, param_dict, dataloader.is_training, dataloader.feature_names)
  features = preprocessor.run()

  # features = librosa.power_to_db(features, ref=np.max)

  # data = np.concatenate([features[img_index,:,:,0], features[img_index,:,:,1], features[img_index,:,:,2], features[img_index,:,:,3],
  #                   features[img_index, :, :, 4], features[img_index, :, :, 5], features[img_index, :, :, 6], features[img_index, :, :, 7],
  #                   features[img_index, :, :, 8], features[img_index, :, :, 9], features[img_index, :, :, 10], features[img_index, :, :, 11],
  #                   features[img_index, :, :, 12]], axis=0)
  #
  # # data = np.concatenate([features[img_index,:,:,3],
  # #                   features[img_index, :, :, 4], features[img_index, :, :, 5], features[img_index, :, :, 6], features[img_index, :, :, 7],
  # #                   features[img_index, :, :, 8], features[img_index, :, :, 9], features[img_index, :, :, 10], features[img_index, :, :, 11]
  # #                        ], axis=0)
  #
  # plt.figure(1)
  # plt.imshow(np.squeeze(data))
  # plt.show()

  fig, axes = plt.subplots(13, 1, figsize=(15, 8))
  fig.subplots_adjust(hspace=.001, wspace=.001)

  axs = axes.ravel()
  for i in range(13):
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].imshow(features[img_index,:, :, i], aspect='auto')
  plt.show()


  temp = 0