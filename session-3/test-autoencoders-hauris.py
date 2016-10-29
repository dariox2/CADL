
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
#from skimage import data
from scipy.misc import imresize
#import IPython.display as ipyd

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


#==============================

# See how this works w/ Celeb Images or try your own dataset instead:

#imgs = ...

#dirname = '/notebooks/CADL/ImagesForProject/ProjectImagesMixed'
#dirname = "../../myrandompictures"
dirname = "../session-1/labdogs"

# Load every image file in the provided directory

filenames = [os.path.join(dirname, fname)

for fname in os.listdir(dirname)  if ".jp" in fname]

# Make sure we have exactly 100 image files!

filenames = filenames[:100]

#print(filenames)

# Read every filename as an RGB image

imgsB = [plt.imread(fname)[..., :3] for fname in filenames]

# Crop every image to a square

imgsB = [utils.imcrop_tosquare(img_i) for img_i in imgsB]

# Then resize the square image to 100 x 100 pixels

imgsB = [resize(img_i, (100, 100)) for img_i in imgsB]

# Finally make our list of 3-D images a 4-D array with the first dimension the number of images:

imgsB = np.array(imgsB).astype(np.float32)

print("imgsB.shape", imgsB.shape)

print("imgsB.dtype", imgsB.dtype)

#plt.figure(figsize=(10, 10))

plt.title("montage")
plt.imshow(utils.montage(imgsB))
plt.pause(1)

Xs = imgsB

print("Xs.shape", Xs.shape)

assert(Xs.ndim == 4 and Xs.shape[1] <= 250 and Xs.shape[2] <= 250)

print("\n\n Xs:\n", Xs)


# POSSIBLE BUGFIX 2
Xs=np.clip(np.array(Xs)*255, 0, 255).astype(np.uint8)

ds = datasets.Dataset(Xs)
#print("\n\n ds.Xs:\n", ds.X)

# ds = datasets.CIFAR10(flatten=False)

# POSSIBLE BUGFIX 1
mean_img = ds.mean().astype(np.uint8)
#mean_img = ds.mean()

print("\n\n mean_img:\n", mean_img)

plt.title("mean")
plt.imshow(mean_img)
plt.pause(3)



std_img = ds.std()
#print(std_img.shape)
print("\n\n std_img:\n", std_img)


plt.title("std dev")
plt.imshow(std_img)
plt.pause(3)




#eop


