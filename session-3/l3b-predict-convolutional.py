
#
# Lecture 3 - Predicting image labels
# Convolutional network
#

import tensorflow as tf
from libs import datasets
import matplotlib.pyplot as plt
import numpy as np

from libs import utils
from libs.utils import montage_filters

import datetime
# dja
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")


# We'll have placeholders just like before which we'll fill in later.
ds = datasets.MNIST(one_hot=True, split=[0.8, 0.1, 0.1])
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

X_tensor = tf.reshape(X, [-1, 28, 28, 1])

n_output = 10

filter_size = 5
n_filters_in = 1
n_filters_out = 32
W_1 = tf.get_variable(
    name='W',
    shape=[filter_size, filter_size, n_filters_in, n_filters_out],
    initializer=tf.random_normal_initializer())


b_1 = tf.get_variable(
    name='b',
    shape=[n_filters_out],
    initializer=tf.constant_initializer())


h_1 = tf.nn.relu(
    tf.nn.bias_add(
        tf.nn.conv2d(input=X_tensor,
                     filter=W_1,
                     strides=[1, 2, 2, 1],
                     padding='SAME'),
        b_1))


n_filters_in = 32
n_filters_out = 64
W_2 = tf.get_variable(
    name='W2',
    shape=[filter_size, filter_size, n_filters_in, n_filters_out],
    initializer=tf.random_normal_initializer())
b_2 = tf.get_variable(
    name='b2',
    shape=[n_filters_out],
    initializer=tf.constant_initializer())
h_2 = tf.nn.relu(
    tf.nn.bias_add(
        tf.nn.conv2d(input=h_1,
                 filter=W_2,
                 strides=[1, 2, 2, 1],
                 padding='SAME'),
        b_2))


# We'll now reshape so we can connect to a fully-connected/linear layer:
h_2_flat = tf.reshape(h_2, [-1, 7 * 7 * n_filters_out])

# NOTE: This uses a slightly different version of the linear function than the lecture!
h_3, W = utils.linear(h_2_flat, 128, activation=tf.nn.relu, name='fc_1')

# NOTE: This uses a slightly different version of the linear function than the lecture!
Y_pred, W = utils.linear(h_3, n_output, activation=tf.nn.softmax, name='fc_2')


print("Begin training...")
t1 = datetime.datetime.now()


cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
sess.run(tf.initialize_all_variables())


batch_size = 50
n_epochs = 10
for epoch_i in range(n_epochs):
    for batch_xs, batch_ys in ds.train.next_batch():
        sess.run(optimizer, feed_dict={
            X: batch_xs,
            Y: batch_ys
        })
    valid = ds.valid
    print(sess.run(accuracy,
                   feed_dict={
                       X: valid.images,
                       Y: valid.labels
                   }))

# Print final test accuracy:
test = ds.test
print("accuracy: ", sess.run(accuracy,
               feed_dict={
                   X: test.images,
                   Y: test.labels
               }))


t2 = datetime.datetime.now()
delta = t2 - t1
print("             Total training time: ", delta.total_seconds())


#from libs.utils import montage_filters
W1 = sess.run(W_1)
plt.figure(figsize=(10, 10))
plt.imshow(montage_filters(W1), cmap='coolwarm', interpolation='nearest')
plt.pause(3)

W2 = sess.run(W_2)
plt.imshow(montage_filters(W2 / np.max(W2)), cmap='coolwarm')


plt.pause(10)
input("press enter...")
plt.close()
# eop




