
#
# Unsupervised learning (lecture)
#

#import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from libs.utils import montage
from libs import gif

import datetime

# dja
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")



from libs.datasets import MNIST
ds = MNIST()

print("ds.X.shape: ", ds.X.shape)

plt.imshow(ds.X[0].reshape((28, 28)))

# Let's get the first 1000 images of the dataset and reshape them
imgs = ds.X[:1000].reshape((-1, 28, 28))

# Then create a montage and draw the montage
plt.imshow(montage(imgs), cmap='gray')
plt.pause(1)

# Take the mean across all images
mean_img = np.mean(ds.X, axis=0)

# Then plot the mean image.
#plt.figure()
plt.imshow(mean_img.reshape((28, 28)), cmap='gray')
plt.title("mean")
plt.pause(1)

# Take the std across all images
std_img = np.std(ds.X, axis=0)

# Then plot the std image.
#plt.figure()
plt.imshow(std_img.reshape((28, 28)))
plt.title("std dev")
plt.pause(1)


#
# CREATE THE NETWORK
#

#
# 1 - Encoder
#

dimensions = [512, 256, 128, 64]

# So the number of features is the second dimension of our inputs matrix, 784
n_features = ds.X.shape[1]

# And we'll create a placeholder in the tensorflow graph that will be able to get any number of n_feature inputs.
X = tf.placeholder(tf.float32, [None, n_features])

# let's first copy our X placeholder to the name current_input
current_input = X
n_input = n_features

# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []

# We'll create a for loop to create each layer:
for layer_i, n_output in enumerate(dimensions):

    # just like in the last session,
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("encoder/layer/{}".format(layer_i)):

        # Create a weight matrix which will increasingly reduce
        # down the amount of information in the input by performing
        # a matrix multiplication
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        # Now we'll multiply our input by our newly created W matrix
        # and add the bias
        h = tf.matmul(current_input, W)
        # wut bias?

        # And then use a relu activation function on its output
        current_input = tf.nn.relu(h)

        # Finally we'll store the weight matrix so we can build the decoder.
        Ws.append(W)

        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output


print("current input shape: ", current_input.get_shape())

#
# 2 - Decoder
#

# We'll first reverse the order of our weight matrices
Ws = Ws[::-1]

# then reverse the order of our dimensions
# appending the last layers number of inputs.
dimensions = dimensions[::-1][1:] + [ds.X.shape[1]]
print("dimensions: ", dimensions)

for layer_i, n_output in enumerate(dimensions):
    # we'll use a variable scope again to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):

        # Now we'll grab the weight matrix we created before and transpose it
        # So a 3072 x 784 matrix would become 784 x 3072
        # or a 256 x 64 matrix, would become 64 x 256
        W = tf.transpose(Ws[layer_i])

        # Now we'll multiply our input by our transposed W matrix
        h = tf.matmul(current_input, W)

        # And then use a relu activation function on its output
        current_input = tf.nn.relu(h)

        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output


Y = current_input

# We'll first measure the average difference across every pixel
cost = tf.reduce_mean(tf.squared_difference(X, Y), 1)
print("cost shape: ", cost.get_shape())

cost = tf.reduce_mean(cost)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# dja
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print("Training...")
t1 = datetime.datetime.now()

# Some parameters for training
batch_size = 100
n_epochs = 10

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
examples = ds.X[:100]

# We'll store the reconstructions in a list
imgs = []
#fig, ax = plt.subplots(1, 1)
for epoch_i in range(n_epochs):
    for batch_X, _ in ds.train.next_batch():
        sess.run(optimizer, feed_dict={X: batch_X - mean_img})
    recon = sess.run(Y, feed_dict={X: examples - mean_img})
    recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
    print("epoch: ", epoch_i, "  cost: ", sess.run(cost, feed_dict={X: batch_X - mean_img}))
    img_i = montage(recon).astype(np.uint8)
    imgs.append(img_i)
    #ax.imshow(img_i, cmap='gray')
    plt.imshow(img_i, cmap='gray')
    plt.title("epoch "+str(epoch_i))
    plt.pause(1)
    #fig.canvas.draw()


t2 = datetime.datetime.now()
delta = t2 - t1
print("             Total training time: ", delta.total_seconds())

gif.build_gif(imgs, saveto='lecture3_mnist_'+TID+'.gif', cmap='gray', interval=0.3, show_gif=False)

plt.pause(10)
input("press enter...")
plt.close()
# eop




