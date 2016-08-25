
#
# test01
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


n_iterations = 100
LAYERSIZE=64
NHIDLAYERS=5

tamimg=64
#filenames=["barvert.png", "barvert.png"]
#filenames=["barhoriz.png", "barhoriz.png"]
#filenames=["barhoriz.png", "barvert.png"]
#filenames=["../../fot2.jpg", "fot1.jpg"]
#filenames=["fot1.jpg", "fot1.jpg"]
filenames=["../../fot2.jpg", "../../fot2.jpg"]

gif_frames=50
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

#dja
plt.ion()
#plt.show()
#plt.pause(2)

origimg = [plt.imread(fname)[..., :3] for fname in filenames] # falla con .jpg?
#origimg=[plt.imread(filenames[0]), plt.imread(filenames[1])]

print(origimg)

imrsz0=imresize(origimg[0], (tamimg,tamimg))
imrsz1=imresize(origimg[1], (tamimg,tamimg))
#print("imrsz0=",imrsz0)
#print("imrsz1=",imrsz1)
scaledimg = [imrsz0, imrsz1]
#print("scaledimg=",scaledimg)

if plotgraph:
  plt.figure(figsize=(5, 5))
  plt.imshow(scaledimg[1])
  plt.title("(preparing the data)")
  plt.show()
  plt.pause(1)
  #plt.close()


#plt.imsave(fname='session2_batch_reference.png', arr=scaledimg)

#print(scaledimg.shape)


xs0, ys0 = split_image(scaledimg[0])
xs1, ys1 = split_image(scaledimg[1])

#print(xs0.__class__)

xs=np.asarray([xs0, xs1])
ys=np.asarray([ys0, ys1])

#print("xs=",xs)
#print("ys=",ys)


#print(xs.__class__)
#print(xs.shape)

#print(xs.shape)

xs=xs.reshape(xs.shape[0]*xs.shape[1], 2)
ys=ys.reshape(ys.shape[0]*ys.shape[1], 3)

print("xs, ys shape:" , xs.shape, ys.shape)

#print("xs=",xs)
#print("ys=",ys)

#print("============ ys:")
#print(ys)

xs = (xs - np.mean(xs)) / np.std(xs)
#print("norm xs: ", xs)

print("norm. x min/max", np.min(xs), np.max(xs))
assert(np.min(xs) > -3.0 and np.max(xs) < 3.0)

# don't look next line
print("y min/max", np.min(ys), np.max(ys))

CLIPVALUE=255
#if np.max(ys)>1.1:  # YA ESTA NORMALIZADO??
#  ys = ys / 255.0
#print("norm. y min/max",np.min(ys), np.max(ys))

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

n_neurons = [2, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE,  3]

print("LAYERSIZE=",LAYERSIZE)

# Input layer
HOUT, W = linear(X, LAYERSIZE, activation=tf.nn.relu, name='Lay1')
# Hidden layers
for i in range(1, NHIDLAYERS+1):
  print("creating hidd lay ", i)
  HOUT, W = linear(HOUT, LAYERSIZE, activation=tf.nn.relu, name='HidLay'+str(i))
# Output layer
Y_pred, W7 = linear(HOUT, 3, activation=None, name='pred')


assert(X.get_shape().as_list() == [None, 2])
assert(Y_pred.get_shape().as_list() == [None, 3])
assert(Y.get_shape().as_list() == [None, 3])

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
step_i = 0
batch_size = int(np.sqrt(len(xs)))
for it_i in range(n_iterations):

    print("iteration: ", it_i, end="", flush=True);
    
    # Get a random sampling of the dataset
    idxs = np.random.permutation(range(len(xs)))
  
    ###print("idxs=",idxs)
    
    # The number of batches we have to iterate over
    n_batches = max(len(idxs) // batch_size, 1)
    #print("  n_batches: ", n_batches, end="", flush=True);

    # Now iterate over our stochastic minibatches:
    for batch_i in range(n_batches):

        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

        #print("===============================")
        #print("xs feed: ", xs[idxs_i])
        #print("ys feed: ", ys[idxs_i])
 
        sess.run(myoptimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

    #OJO, indent
    training_cost = sess.run(costtot, feed_dict={X: xs, Y: ys})

    #print("  cost: ", training_cost / n_batches);
    print("  cost: ", training_cost);

    if (it_i + 1) % gif_step == 0 or (it_i + 1) % plot_step == 0:
        idxs_j=range(len(xs)//2)
        ys_pred = Y_pred.eval(feed_dict={X: xs[idxs_j]}, session=sess)
        # PARA PNG:
        #plotimg = np.clip(np.array(ys_pred.reshape(scaledimg[0].shape))*255, 0, CLIPVALUE).astype(np.uint8)
        # PARA JPG:
        plotimg = np.clip(np.array(ys_pred.reshape(scaledimg[0].shape)), 0, CLIPVALUE).astype(np.uint8)
    if (it_i + 1) % gif_step == 0:
        gifimgs.append(plotimg)
    if (it_i + 1) % plot_step == 0:
        costs.append(training_cost)
        if plotgraph:
          plt.imshow(plotimg)
          plt.title('Iteration {}'.format(it_i))
          plt.show()
          plt.pause(1)

#print(ys_pred)
  
if plotgraph:
  # Save the images as a GIF
  _ = gif.build_gif(gifimgs, saveto='test01_single.gif', show_gif=False)

  plt.imsave(fname='test01_predicted.png', arr=plotimg, interval=0.3)

  plt.pause(5)
  plt.close()


# eop


