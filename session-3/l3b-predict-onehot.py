
#
# Lecture 3 - Predicting image labels
# One-hot encoding
#

#import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from libs.utils import montage
from libs import gif

# dja
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()

from libs import datasets
# ds = datasets.MNIST(one_hot=True)

ds = datasets.MNIST(one_hot=False)
# let's look at the first label
print(ds.Y[0])
# okay and what does the input look like
plt.imshow(np.reshape(ds.X[0], (28, 28)), cmap='gray')
# great it is just the label of the image

plt.figure()
# Let's look at the next one just to be sure
print(ds.Y[1])
# Yea the same idea
plt.imshow(np.reshape(ds.X[1], (28, 28)), cmap='gray')

ds = datasets.MNIST(one_hot=True)
plt.figure()
plt.imshow(np.reshape(ds.X[0], (28, 28)), cmap='gray')
print(ds.Y[0])
# array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.])
# Woah a bunch more numbers.  10 to be exact, which is also the number
# of different labels in the dataset.
plt.imshow(np.reshape(ds.X[1], (28, 28)), cmap='gray')
print(ds.Y[1])
# array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.])

print(ds.X.shape)


plt.pause(10)
input("press enter...")
plt.close()
# eop


