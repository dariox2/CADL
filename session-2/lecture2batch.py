
#
# Training a Network w/ Tensorflow (lecture)
#

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from libs import gif

#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)


# this function will measure the absolute distance, also known as the l1-norm
def distance(p1, p2):
    return tf.abs(p1 - p2)


def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h


#dja
plt.ion()


from skimage.data import astronaut
from scipy.misc import imresize
img = imresize(astronaut(), (64, 64))
#img = plt.imread("mypictures/mediumtree.jpg")

plt.imsave(fname='lecture2_batch_reference.png', arr=img)

#plt.imshow(img)
#plt.show()

gifimgs = []

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

print("========ys:")
print(ys)

# Normalizing the input by the mean and standard deviation
xs = (xs - np.mean(xs)) / np.std(xs)

# and print the shapes
xs.shape, ys.shape

plt.imshow(ys.reshape(img.shape))
plt.title("(reshaped from array)")
plt.show()
plt.pause(1)
#plt.close()

X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]

current_input = X
for layer_i in range(1, len(n_neurons)):
    current_input = linear(
        X=current_input,
        n_input=n_neurons[layer_i - 1],
        n_output=n_neurons[layer_i],
        activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,
        scope='layer_' + str(layer_i))
Y_pred = current_input


cost = tf.reduce_mean(
    tf.reduce_sum(distance(Y_pred, Y), 1))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)



n_iterations = 500
batch_size = 50

#fig, ax = plt.subplots(1, 1)

with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    # This will set W and b to their initial random normal value.
    sess.run(tf.initialize_all_variables())

    # We now run a loop over epochs
    prev_training_cost = 0.0
    for it_i in range(n_iterations):
        idxs = np.random.permutation(range(len(xs)))
        n_batches = len(idxs) // batch_size
        print("  n_batches: ", n_batches, end="", flush=True);
        for batch_i in range(n_batches):
            idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

        training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
        print("  it: ", it_i, "  cost: ", training_cost)

        if (it_i + 1) % 20 == 0:
            ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
            print("============ ys_pred:")
            print(np.floor(ys_pred))
            #fig, ax = plt.subplots(1, 1)
            img = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)

            gifimgs.append(img)

            plt.title('Iteration {}'.format(it_i))
            plt.imshow(img)
            plt.show()
            plt.pause(1)

# Save the images as a GIF
_ = gif.build_gif(gifimgs, saveto='lecture2_batch_single.gif', show_gif=False)

plt.imsave(fname='lecture2_batch_predicted.png', arr=img)

plt.pause(10)
plt.close()



