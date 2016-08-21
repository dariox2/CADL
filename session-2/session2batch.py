
#
# Training a Network w/ Tensorflow
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


#dja
plotgraph=True

#plt.style.use('ggplot')
plt.style.use('bmh')

#dja
plt.ion()
#plt.show()
#plt.pause(2)

origimg = plt.imread("mypictures/tux-small.jpg")
# = plt.imread("mypictures/tux-large.jpg")
#origimg = plt.imread("mypictures/mediumtree.jpg")
#from skimage.data import astronaut
#from scipy.misc import imresize
#origimg = imresize(astronaut(), (64, 64))


scaledimg = imresize(origimg, (64,64))
#scaledimg=origimg
if plotgraph:
  plt.figure(figsize=(5, 5))
  plt.imshow(scaledimg)
  plt.title("(preparing the data)")
  plt.show()
  plt.pause(1)
  #plt.close()


plt.imsave(fname='session2_batch_reference.png', arr=scaledimg)

print(scaledimg.shape)

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

xs, ys = split_image(scaledimg)
print("x, y shape:" , xs.shape, ys.shape)

print("============ ys:")
print(ys)

xs = (xs - np.mean(xs)) / np.std(xs)
print("norm xs: ", xs)

print("norm. x min/max", np.min(xs), np.max(xs))
assert(np.min(xs) > -3.0 and np.max(xs) < 3.0)


print("y min/max", np.min(ys), np.max(ys))

CLIPVALUE=255
#if np.max(ys)>1.1:  # YA ESTA NORMALIZADO??
#  ys = ys / 255.0
print("norm. y min/max",np.min(ys), np.max(ys))

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

LAYERSIZE=64
n_neurons = [2, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE,  3]

print("LAYERSIZE=",LAYERSIZE)

h1, W1 = linear(X, LAYERSIZE, activation=tf.nn.relu, name='Lay1')
#
h2, W2 = linear(h1, LAYERSIZE, activation=tf.nn.relu, name='Lay2')
h3, W3 = linear(h2, LAYERSIZE, activation=tf.nn.relu, name='Lay3')
h4, W4 = linear(h3, LAYERSIZE, activation=tf.nn.relu, name='Lay4')
h5, W5 = linear(h4, LAYERSIZE, activation=tf.nn.relu, name='Lay5')
h6, W6 = linear(h5, LAYERSIZE, activation=tf.nn.relu, name='Lay6')
#
Y_pred, W7 = linear(h6, 3, activation=None, name='pred')


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
n_iterations = 500
batch_size = 50

sess = tf.Session()

sess.run(tf.initialize_all_variables())

gifimgs = []
costs = []
#gif_step = n_iterations // 10
gif_step=20
print("gif_step: ", gif_step)
step_i = 0

for it_i in range(n_iterations):

    print("iteration: ", it_i, end="", flush=True);
    
    # Get a random sampling of the dataset
    idxs = np.random.permutation(range(len(xs)))
    
    # The number of batches we have to iterate over
    n_batches = max(len(idxs) // batch_size, 1)
    print("  n_batches: ", n_batches, end="", flush=True);

    # Now iterate over our stochastic minibatches:
    for batch_i in range(n_batches):

        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

        #training_cost = sess.run([costtot, myoptimizer],feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]
        sess.run(myoptimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})
    #OJO, indent
    training_cost = sess.run(costtot, feed_dict={X: xs, Y: ys})

    #print("  cost: ", training_cost / n_batches);
    print("  cost: ", training_cost);

    if (it_i + 1) % gif_step == 0:
        #costs.append(training_cost / n_batches)
        costs.append(training_cost)
        ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
        print( "ys_pred shape: ", ys_pred.shape)
        print("============ ys_pred:")
        print(np.floor(ys_pred))
        if plotgraph:
          ###plotimg = np.clip(ys_pred.reshape(scaledimg.shape), 0, CLIPVALUE)
          plotimg = np.clip(ys_pred.reshape(scaledimg.shape), 0, CLIPVALUE).astype(np.uint8)
          gifimgs.append(plotimg)
          # Plot the cost over time
          #fig, ax = plt.subplots(1, 2)
          plt.imshow(plotimg)
          plt.title('Iteration {}'.format(it_i))
          plt.show()
          plt.pause(1)
  
if plotgraph:
  # Save the images as a GIF
  _ = gif.build_gif(gifimgs, saveto='session2_batch_single.gif', show_gif=False)

  plt.imsave(fname='session2_batch_predicted.png', arr=plotimg)

  plt.pause(5)
  plt.close()


# eop


