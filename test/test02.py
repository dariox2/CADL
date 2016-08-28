
#
# test02
#
# morphing
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import tensorflow as tf
from libs import gif
import IPython.display as ipyd


n_iterations=20
LAYERSIZE=16
NHIDLAYERS=5

tamimg=64
#filenames=["barvert.png", "barvert.png"]
#filenames=["barhoriz.png", "barhoriz.png"]
filenames=["barhoriz.png", "barvert.png"]
#filenames=["../../fot2.jpg", "fot1.jpg"]

gif_frames=20
plot_step=1

#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)

# (from utils.py)
def linear(x, n_output, name=None, activation=None, reuse=None):
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            #initializer=tf.contrib.layers.xavier_initializer())
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            #initializer=tf.constant_initializer(0.0))
            initializer=tf.constant_initializer())

        #h = tf.nn.bias_add(
        #    name='h',
        #    value=tf.matmul(x, W),
        #    bias=b)
        #if activation:
        #    h = activation(h)

        h = tf.matmul(x, W) + b
        if activation is not None: # esta linea da error: 'Tensor' object is not iterable
            h = activation(h)
        # return h

        return h, W


def split_image(img):
    xs = []

    ys = []

    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])

    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


#########################################
#
# MAIN
#


#dja
plotgraph=True

#plt.style.use('ggplot')
plt.style.use('bmh')

plt.ion()

origimg = [plt.imread(fname)[..., :3] for fname in filenames] 

imrsz0=imresize(origimg[0], (tamimg,tamimg))
imrsz1=imresize(origimg[1], (tamimg,tamimg))
scaledimg=[imrsz0, imrsz1]

if plotgraph:
  plt.figure(figsize=(5, 5))
  meanimg=((imrsz0+imrsz1)/2).astype(np.uint8)
  plt.imshow(meanimg)

  plt.title("(preparing the data)")
  plt.show()
  plt.pause(10)
  #plt.close()


xs0, ys0 = split_image(scaledimg[0])
xs1, ys1 = split_image(scaledimg[1])

xs0 = (xs0 - np.mean(xs0)) / np.std(xs0)
xs1 = xs0

CLIPVALUE=255

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

print("LAYERSIZE=",LAYERSIZE)

# Input layer
HOUT, W = linear(X, LAYERSIZE, activation=tf.nn.relu, name='Lay1')
# Hidden layers
for i in range(1, NHIDLAYERS+1):
  print("creating hidd lay ", i)
  HOUT, W = linear(HOUT, LAYERSIZE, activation=tf.nn.relu, name='HidLay'+str(i))
# Output layer
Y_pred, W7 = linear(HOUT, 3, activation=None, name='pred')


#errortot = tf.abs(Y_pred - Y)
#errortot = tf.pow(tf.sub(Y_pred, Y), 2)
errortot = (Y_pred - Y) ** 2
assert(errortot.get_shape().as_list() == [None, 3])
print("error.shape: ", errortot.get_shape())

sum_errorred = tf.reduce_sum(errortot, 1)
assert(sum_errorred.get_shape().as_list() == [None])

costtot = tf.reduce_mean(sum_errorred)
assert(costtot.get_shape().as_list() == [])

myoptimizer =tf.train.AdamOptimizer(0.001).minimize(costtot)

sess = tf.Session()

sess.run(tf.initialize_all_variables())

gifimgs = []
costs = []
gif_step = max(n_iterations // gif_frames, 1)
print("gif_step: ", gif_step)
batch_size = int(np.sqrt(len(xs0)))
for it_i in range(1, (n_iterations)*3+1):

    cicl=it_i//n_iterations;

    print("iteration: ", it_i, " cicl: ", cicl, end="", flush=True);
    
    # Get a random sampling of the dataset
    idxs = np.random.permutation(range(len(xs0)))
  
    # The number of batches we have to iterate over
    n_batches = max(len(idxs) // batch_size, 1)
    #print("  n_batches: ", n_batches, end="", flush=True);

    # Now iterate over our stochastic minibatches:
    for batch_i in range(n_batches):

        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

        if (cicl)==1:
          sess.run(myoptimizer, feed_dict={X: xs0[idxs_i], Y: ys0[idxs_i]})
        else: # 0, 2
          sess.run(myoptimizer, feed_dict={X: xs1[idxs_i], Y: ys1[idxs_i]})


    #OJO, indent
    if (cicl)==1:
      training_cost = sess.run(costtot, feed_dict={X: xs0, Y: ys0})
    else:
      training_cost = sess.run(costtot, feed_dict={X: xs1, Y: ys1})

    #print("  cost: ", training_cost / n_batches);
    print("  cost: ", training_cost);

    if (it_i % gif_step) == 0 or (it_i % plot_step) == 0:
        idxs_j=range(len(xs0))
        ys_pred = Y_pred.eval(feed_dict={X: xs0[idxs_j]}, session=sess)
        # FIXED (probado jpg/png):
        plotimg = np.clip(np.array(ys_pred.reshape(scaledimg[0].shape)), 0, CLIPVALUE).astype(np.uint8)
    if (it_i % gif_step) == 0 and cicl>0:
        gifimgs.append(plotimg)
    if (it_i % plot_step) == 0:
        costs.append(training_cost)
        if plotgraph:
          plt.imshow(plotimg)
          plt.title('Iteration {}'.format(it_i))
          plt.show()
          plt.pause(1)

#print(ys_pred)
  
if plotgraph:
  # Save the images as a GIF
  _ = gif.build_gif(gifimgs, saveto='test02_single.gif', show_gif=False, interval=0.3)

  plt.pause(5)
  plt.close()


# eop


