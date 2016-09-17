
#
# Lecture 3 - Predicting image labels
# Fully connected network
# (includes one-hot encoding, cross entropy)
#

import tensorflow as tf
from libs import datasets
import matplotlib.pyplot as plt
import numpy as np

import datetime
# dja
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")

ds = datasets.MNIST(split=[0.8, 0.1, 0.1])
n_input = 28 * 28

n_output = 10

X = tf.placeholder(tf.float32, [None, n_input])

Y = tf.placeholder(tf.float32, [None, n_output])

# We'll use the linear layer we created in the last session, which I've stored in the libs file:
# NOTE: The lecture used an older version of this function which had a slightly different definition.
from libs import utils
Y_pred, W = utils.linear(
    x=X,
    n_output=n_output,
    activation=tf.nn.softmax,
    name='layer1')

# We add 1e-12 because the log is undefined at 0.
cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)

correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


print("Begin training...")
t1 = datetime.datetime.now()


sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Now actually do some training:
batch_size = 50
n_epochs = 5
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
print("final accuracy: ", sess.run(accuracy,
               feed_dict={
                   X: test.images,
                   Y: test.labels
               }))


t2 = datetime.datetime.now()
delta = t2 - t1
print("             Total training time: ", delta.total_seconds())


#
# Inspecting the trained network
#

# We first get the graph that we used to compute the network
g = tf.get_default_graph()

# And can inspect everything inside of it
##[op.name for op in g.get_operations()]
#for op in g.get_operations():
#  print(op.name)

W = g.get_tensor_by_name('layer1/W:0')

W_arr = np.array(W.eval(session=sess))
print(W_arr.shape)

fig, ax = plt.subplots(1, 10, figsize=(20, 3))
for col_i in range(10):
    ax[col_i].imshow(W_arr[:, col_i].reshape((28, 28)), cmap='coolwarm')



plt.pause(10)
input("press enter...")
plt.close()
# eop




