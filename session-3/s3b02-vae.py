
#
# Session 3, part 2
#
# VAE
#

import sys

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import IPython.display as ipyd

print("Loading tensorflow...")
import tensorflow as tf

from libs import utils, gif, datasets, dataset_utils, vae, dft

#plt.style.use('ggplot')
plt.style.use('bmh')

import datetime

# dja
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
plt.figure(figsize=(3, 3))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")


print("Loading pictures...")

#some_dir = "../session-1/labdogs"
some_dir = "../../myrandompictures"
input_shape=[100,100,3]

# Get a list of jpg file (Only JPG works!)
files = [os.path.join(some_dir, file_i)
         for file_i in os.listdir(some_dir)
             if (file_i.endswith('.jpg') or file_i.endswith('.jpeg') or file_i.endswith('.png'))]

print("files:")
print(files)

print("Training...")
t1 = datetime.datetime.now()


# Train it!  Change these parameters!
vae.train_vae(files,
              input_shape,
              learning_rate=0.0001,
              batch_size=10,
              n_epochs=40,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=20,
              save_step=20,
              ckpt_name="vae2.ckpt")


t2 = datetime.datetime.now()
delta = t2 - t1
print("             Total training time: ", delta.total_seconds())


#eop


