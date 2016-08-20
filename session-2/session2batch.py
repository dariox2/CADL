
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
from libs import utils, gif
import IPython.display as ipyd


#dja
plotgraph=True

# We'll tell matplotlib to inline any drawn figures like so:
###%matplotlib inline

#plt.style.use('ggplot')
# ['seaborn-bright', 'seaborn-darkgrid', 'seaborn-ticks', 'classic', 'fivethirtyeight', 'dark_background', 'ggplot', 'seaborn-white', 'seaborn-paper', 'seaborn-poster', 'seaborn-pastel', 'seaborn-talk', 'seaborn-notebook', 'seaborn-dark', 'seaborn-deep', 'grayscale', 'bmh', 'seaborn-colorblind', 'seaborn-whitegrid', 'seaborn-muted', 'seaborn-dark-palette']
plt.style.use('bmh')


#--2
# Bit of formatting because I don't like the default inline code style:

#
# Part One - Fully Connected Network
#


#dja
plt.ion()
#plt.show()
#plt.pause(2)


#
# Part Two - Image Painting Network
#

#
# Preparing the Data
#


# TODO! COMPLETE THIS SECTION!
# First load an image
import matplotlib.pyplot as plt
#origimg = plt.imread("mypictures/tux-small.jpg")
# = plt.imread("mypictures/tux-large.jpg")
origimg = plt.imread("mypictures/mediumtree.jpg")
#
# Be careful with the size of your image.
# Try a fairly small image to begin with,
# then come back here and try larger sizes.

scaledimg = imresize(origimg, (64,64))
#scaledimg=origimg
if plotgraph:
  plt.figure(figsize=(5, 5))
  plt.imshow(scaledimg)
  plt.title("(preparing the data)")
  plt.show()
  plt.pause(3)
  plt.close()

#
# Make sure you save this image as "reference.png"
# and include it in your zipped submission file
# so we can tell what image you are trying to paint!
plt.imsave(fname='reference_batch.png', arr=scaledimg)


print(scaledimg.shape)

def split_image(img):
    # We'll first collect all the positions in the image in our list, xs
    xs = []

    # And the corresponding colors for each of these positions
    ys = []

    # Now loop over the image
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            # And store the inputs
            xs.append([row_i, col_i])
            # And outputs that the network needs to learn to predict
            ys.append(img[row_i, col_i])

    # we'll convert our lists to arrays
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

xs, ys = split_image(scaledimg)
# and print the shapes
print("x, y shape:" , xs.shape, ys.shape)

print("xs: ", xs)
print("ys: ", ys)

# TODO! COMPLETE THIS SECTION!
# Normalize the input (xs) using its mean and standard deviation
xs = (xs - np.mean(xs)) / np.std(xs)
print("norm xs: ", xs)

#
# Just to make sure you have normalized it correctly:
print("norm. x min/max", np.min(xs), np.max(xs))
assert(np.min(xs) > -3.0 and np.max(xs) < 3.0)


print("y min/max", np.min(ys), np.max(ys))

CLIPVALUE=255
#if np.max(ys)>1.1:  # YA ESTA NORMALIZADO??
#  ys = ys / 255.0
print("norm. y min/max",np.min(ys), np.max(ys))


# TODO! COMPLETE THIS SECTION!
# Let's reset the graph:
tf.reset_default_graph()
#
# Create a placeholder of None x 2 dimensions and dtype tf.float32
# This will be the input to the network which takes the row/col
X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
#
# Create the placeholder, Y, with 3 output dimensions instead of 2.
# This will be the output of the network, the R, G, B values.
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')


# TODO! COMPLETE THIS SECTION!
# We'll create 6 hidden layers.  Let's create a variable
# to say how many neurons we want for each of the layers
# (try 20 to begin with, then explore other values)
LAYERSIZE=64
n_neurons = [2, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE, LAYERSIZE,  3]
#
# Create the first linear + nonlinear layer which will
# take the 2 input neurons and fully connects it to 20 neurons.
# Use the `utils.linear` function to do this just like before,
# but also remember to give names for each layer, such as
# "1", "2", ... "5", or "layer1", "layer2", ... "layer6".
h1, W1 = utils.linear(X, LAYERSIZE, activation=None, name='Lay1')
#
# Create another one:
h2, W2 = utils.linear(h1, LAYERSIZE, activation=None, name='Lay2')
#
# and four more (or replace all of this with a loop if you can!):
h3, W3 = utils.linear(h2, LAYERSIZE, activation=None, name='Lay3')
h4, W4 = utils.linear(h3, LAYERSIZE, activation=None, name='Lay4')
h5, W5 = utils.linear(h4, LAYERSIZE, activation=None, name='Lay5')
h6, W6 = utils.linear(h5, LAYERSIZE, activation=None, name='Lay6')
#
# Now, make one last layer to make sure your network has 3 outputs:
Y_pred, W7 = utils.linear(h6, 3, activation=None, name='pred')


assert(X.get_shape().as_list() == [None, 2])
assert(Y_pred.get_shape().as_list() == [None, 3])
assert(Y.get_shape().as_list() == [None, 3])


#-- hasta aqui todo ok


# TODO! COMPLETE THIS SECTION!
# first compute the error, the inner part of the summation.
# This should be the l1-norm or l2-norm of the distance
# between each color channel.
#error = tf.square(tf.sub(Y, Y_pred))
error = tf.pow(tf.abs(tf.sub(Y_pred, Y)), 2)
assert(error.get_shape().as_list() == [None, 3])
print("error.shape: ", error.get_shape())


# TODO! COMPLETE THIS SECTION!
# Now sum the error for each feature in Y. 
# If Y is [Batch, Features], the sum should be [Batch]:
sum_error = tf.reduce_sum(error, 1)
assert(sum_error.get_shape().as_list() == [None])


# TODO! COMPLETE THIS SECTION!
# Finally, compute the cost, as the mean error of the batch.
# This should be a single value.
cost = tf.reduce_mean(sum_error)
assert(cost.get_shape().as_list() == [])


# TODO! COMPLETE THIS SECTION!
# Refer to the help for the function
optimizer =tf.train.AdamOptimizer(0.001).minimize(cost)
#optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
#
# Create parameters for the number of iterations to run for (< 100)
n_iterations = 100
#
# And how much data is in each minibatch (< 500)
batch_size = 50
#
# Then create a session
sess = tf.Session()


# Initialize all your variables and run the operation with your session
sess.run(tf.initialize_all_variables())

# Optimize over a few iterations, each time following the gradient
# a little at a time
gifimgs = []
costs = []
gif_step = n_iterations // 10
print("gif_step: ", gif_step)
step_i = 0

if plotgraph:
  fig, ax = plt.subplots(1, 2)

for it_i in range(n_iterations):

    print("iteration: ", it_i, end="", flush=True);
    
    # Get a random sampling of the dataset
    idxs = np.random.permutation(range(len(xs)))
    
    # The number of batches we have to iterate over
    n_batches = max(len(idxs) // batch_size, 1)
    print("  n_batches: ", n_batches, end="", flush=True);

    # Now iterate over our stochastic minibatches:
    for batch_i in range(n_batches):

        #print(batch_i, end="", flush=True)
         
        # Get just minibatch amount of data
        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

        # And optimize, also returning the cost so we can monitor
        # how our optimization is doing.
        training_cost = sess.run(
            [cost, optimizer],
            feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]

    print("  cost: ", training_cost / n_batches);

    # Also, every 20 iterations, we'll draw the prediction of our
    # input xs, which should try to recreate our image!
    if (it_i + 1) % gif_step == 0:
        costs.append(training_cost / n_batches)
        ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
        print("ys_pred: ", ys_pred, " ys_pred shape: ", ys_pred.shape)
        if plotgraph:
          plotimg = np.clip(ys_pred.reshape(scaledimg.shape), 0, CLIPVALUE)
          #plotimg = ys_pred.reshape(img.shape)
          gifimgs.append(plotimg)
          # Plot the cost over time
          #fig, ax = plt.subplots(1, 2)
          ax[0].plot(costs)
          ax[0].set_xlabel('Iteration')
          ax[0].set_ylabel('Cost')
          ax[1].imshow(plotimg)
          fig.suptitle('Iteration {}'.format(it_i))
          plt.show()
          plt.pause(1)
  
if plotgraph:
  # Save the images as a GIF
  _ = gif.build_gif(gifimgs, saveto='single_batch.gif', show_gif=False)

  plt.pause(10)
  plt.close()


# eop


