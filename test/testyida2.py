

print("Loading tensorflow...")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from libs import utils
import datetime


tf.set_random_seed(1)


def CELEByida(path='./img_align_celeba/'):
  fl=os.listdir(path)
  i=0
  fs=[]
  for f in fl:
       if f.endswith('.jpg'):
         #print(i,": ",f)
         fs.append(os.path.join(path, f))
         i+=1
 
  fs=sorted(fs)
  return fs



print("Loading celebrities...")
from libs.datasets import CELEB
files = CELEByida("../session-1/img_align_celeba/") # only 100

from libs.dataset_utils import create_input_pipeline
batch_size = 9
n_epochs = 3
input_shape = [218, 178, 3]
crop_shape = [64, 64, 3]
crop_factor = 0.8

seed=15

batch = create_input_pipeline(  
    files=files,
    batch_size=batch_size,
    n_epochs=n_epochs,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape,
    n_threads=1) # critical for repeating results

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

batch_xs = sess.run(batch)

mntg=[]
for i in range(0,len(batch_xs)):
  img= batch_xs[i] / 255.0
  mntg.append(img)

TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")
m=utils.montage(mntg, saveto="montage_"+TID+".png")

plt.figure(figsize=(5, 5))
plt.imshow(m)
plt.show()

# eop

