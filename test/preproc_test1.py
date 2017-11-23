from dataloader import DataLoader
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

# training_loader = DataLoader('dataset/audio_list.csv', 'dataset/labels.pkl', 32, val_set_number = 0, is_training=True)
# track_features, label_onehot, titles = training_loader.next_batch()
# track_features : [32, 20, 1723, 1]
# label_onehot : [32, 3, 200]

preproc_dic = {'gamma_mel':0.5, 'norm_type':2, 'target_height':8, 'max_noise':1.0}
dataloader = DataLoader(preprocess_args=preproc_dic)
features, labels, titles = dataloader.next_batch()

data = np.concatenate([features[0,:,:,0], features[0,:,:,1], features[0,:,:,2], features[0,:,:,3],
                  features[0, :, :, 4], features[0, :, :, 5], features[0, :, :, 6], features[0, :, :, 7],
                  features[0, :, :, 8], features[0, :, :, 9], features[0, :, :, 10], features[0, :, :, 11],
                  features[0, :, :, 12]], axis=0)

plt.figure(1)
plt.imshow(np.squeeze(data))
plt.show()

temp = 0