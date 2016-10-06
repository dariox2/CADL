
#
# testsancho
#
# test saving training checkpoint
# CHECKPOINT DOES NOT WORK, 
# train. saver() does not save what  is 
# supposed to (i.e. cell state variables)
#
# sancho2a - 2) only restore
#

print("Loading tensorflow...")
import tensorflow as tf
import numpy as np
import os

# dja
#import os
import datetime
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")


print("Reading text file...")

#f = 'alice.txt'
#f="abece.txt"
f="bohemian.txt" # batch 20, seq 30, 200 it, funny resemblance
with open(f, 'r') as fp:
    txt = fp.read()

runlimit=100

# And let's find out what's inside this text file by creating a set
# of all possible characters.

vocab = list(set(txt))
print ("txt: ", len(txt), "  vocab: ", len(vocab))


# Great so we now have about 164 thousand characters and 85 unique
# characters in our vocabulary which we can use to help us train a
# model of language. Rather than use the characters, we'll convert
# each character to a unique integer. We'll later see that when we
# work with words, we can achieve a similar goal using a very popular
# model called word2vec:
# https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html
#
# We'll first create a look up table which will map a character to an
# integer:

print("Creating encoder...")
encoder = dict(zip(vocab, range(len(vocab))))
print("Creating decoder...")
decoder = dict(zip(range(len(vocab)), vocab))


#
# Creating the Model
#

# For our model, we'll need to define a few parameters.

# Number of sequences in a mini batch
batch_size = 20#0

# Number of characters in a sequence
sequence_length = 30#0

# Number of cells in our LSTM layer
n_cells = 256

# Number of LSTM layers
n_layers = 2

# Total number of characters in the one-hot encoding
n_chars = len(vocab)


# Now create the input and output to the network. Rather than having
# `batch size` x `number of features`; or `batch size` x `height` x
# `width` x `channels`; we're going to have `batch size` x `sequence
# length`.

X = tf.placeholder(tf.int32, [None, sequence_length], name='X')

# We'll have a placeholder for our true outputs
Y = tf.placeholder(tf.int32, [None, sequence_length], name='Y')


# Now remember with MNIST that we used a one-hot vector
# representation of our numbers. We could transform our input data
# into such a representation. But instead, we'll use
# `tf.nn.embedding_lookup` so that we don't need to compute the
# encoded vector. Let's see how this works:

# we first create a variable to take us from our one-hot
# representation to our LSTM cells
embedding = tf.get_variable("embedding", [n_chars, n_cells])

# And then use tensorflow's embedding lookup to look up the ids in X
Xs = tf.nn.embedding_lookup(embedding, X)

# The resulting lookups are concatenated into a dense tensor
print("Xs.get_shape: ", Xs.get_shape().as_list())


# To create a recurrent network, we're going to need to slice our
# sequences into individual inputs. That will give us timestep lists
# which are each `batch_size` x `input_size`. Each character will
# then be connected to a recurrent layer composed of `n_cells` LSTM
# units.

# Let's create a name scope for the operations to clean things up in
# our graph
with tf.name_scope('reslice'):
    Xs = [tf.squeeze(seq, [1])
          for seq in tf.split(1, sequence_length, Xs)]


# Now we'll create our recurrent layer composed of LSTM cells.

cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_cells, state_is_tuple=True)


# We'll initialize our LSTMs using the convenience method provided by
# tensorflow. We could explicitly define the batch size here or use
# the `tf.shape` method to compute it based on whatever `X` is,
# letting us feed in different sizes into the graph.

print("tf.shape(X): ", tf.shape(X))
initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
print("initial state: ", initial_state)
input
# Great now we have a layer of recurrent cells and a way to
# initialize them. If we wanted to make this a multi-layer recurrent
# network, we could use the `MultiRNNCell` like so:

#if n_layers > 1:
#    cells = tf.nn.rnn_cell.MultiRNNCell(
#        [cells] * n_layers, state_is_tuple=True)
#    initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)


# In either case, the cells are composed of their outputs as
# modulated by the LSTM's output gate, and whatever is currently
# stored in its memory contents. Now let's connect our input to it.

# this will return us a list of outputs of every element in our
# sequence.
# Each output is `batch_size` x `n_cells` of output.
# It will also return the state as a tuple of the n_cells's memory
# and their output to connect to the time we use the recurrent layer.
outputs, state = tf.nn.rnn(cells, Xs, initial_state=initial_state)

# We'll now stack all our outputs for every cell
outputs_flat = tf.reshape(tf.concat(1, outputs), [-1, n_cells])


# For our output, we'll simply try to predict the very next timestep.
# So if our input sequence was "networ", our output sequence should
# be: "etwork". This will give us the same batch size coming out, and
# the same number of elements as our input sequence.

print("Creating prediction layer...")
with tf.variable_scope('prediction'):
    W = tf.get_variable(
        "W",
        shape=[n_cells, n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable(
        "b",
        shape=[n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))

    # Find the output prediction of every single character in our minibatch
    # we denote the pre-activation prediction, logits.
    logits = tf.matmul(outputs_flat, W) + b

    # We get the probabilistic version by calculating the softmax of this
    probs = tf.nn.softmax(logits)

    # And then we can find the index of maximum probability
    #Y_pred = tf.argmax(probs)
    Y_pred = tf.argmax(probs, 1)

#
# Loss
#

# Our loss function will take the reshaped predictions and targets,
# and compute the softmax cross entropy.

print("Creating loss function...")
with tf.variable_scope('loss'):
    # Compute mean cross entropy loss for each output.
    Y_true_flat = tf.reshape(tf.concat(1, Y), [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y_true_flat)
    mean_loss = tf.reduce_mean(loss)


#
# Clipping the Gradient
#

# Normally, we would just create an optimizer, give it a learning
# rate, and tell it to minize our loss. But with recurrent networks,
# we can help out a bit by telling it to clip gradients. That helps
# with the exploding gradient problem, ensureing they can't get any
# bigger than the value we tell it. We can do that in tensorflow by
# iterating over every gradient and variable, and changing their
# value before we apply their update to every trainable variable.

print("Creating optimizer & clip gradients...")
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    gradients = []
    clip = tf.constant(5.0, name="clip")
    for grad, var in optimizer.compute_gradients(mean_loss):
        gradients.append((tf.clip_by_value(grad, -clip, clip), var))
    updates = optimizer.apply_gradients(gradients)

# We could also explore other methods of clipping the gradient based
# on a percentile of the norm of activations or other similar
# methods, like when we explored deep dream regularization. But the
# LSTM has been built to help regularize the network through its own
# gating mechanisms, so this may not be the best idea for your
# problem. Really, the only way to know is to try different
# approaches and see how it effects the output on your problem.
#


#
# Training
#

sess = tf.Session()

cursor = 0
it_i = 0


ckptname="testsancho2_model.ckpt-19"
#metaname=ckptname+".meta"

#if os.path.exists(metaname):
if os.path.exists(ckptname):
    #saver = tf.train.import_meta_graph(metaname)
    saver=tf.train.Saver()
    print("Restoring model checkpoint...")
    saver.restore(sess, ckptname)
    print("  Model restored.")

    #ypc=tf.get_collection("Y_pred")
    #for n,t in enumerate(ypc):
    #  print(n, t)
    #Y_pred=ypc[0]
   
    print("  Initializing...")    
    #init = tf.initialize_all_variables()
    sess.run(init)

else:
    print("No checkpoint found")
    quit()

print("Train size: ", batch_size*sequence_length)
print("Begin training...")
#while True:
while it_i<runlimit:
    Xs, Ys = [], []
    for batch_i in range(batch_size):
        if (cursor + sequence_length) >= len(txt) - sequence_length - 1:
            cursor = 0
        Xs.append([encoder[ch]
                   for ch in txt[cursor:cursor + sequence_length]])
        Ys.append([encoder[ch]
                   for ch in txt[cursor + 1: cursor + sequence_length + 1]])

        cursor = (cursor + sequence_length)
    Xs = np.array(Xs).astype(np.int32)
    Ys = np.array(Ys).astype(np.int32)

    if it_i == 0:
        print("PREDICCION INICIAL")
        p = sess.run([Y_pred], feed_dict={X: Xs})[0]
        preds = [decoder[p_i] for p_i in p]
        print("".join(preds).split('\n'))
        print("")
 
    loss_val = sess.run([mean_loss],
                           feed_dict={X: Xs, Y: Ys})

    if it_i % 10 == 0:
        print("it_i: ", it_i, "  loss: ", loss_val)
        p = sess.run([Y_pred], feed_dict={X: Xs})[0]
        preds = [decoder[p_i] for p_i in p]
        print("".join(preds).split('\n'))
        print("")

    it_i += 1

    #if it_i % 50 == 0:
    #    print("Saving checkpoint...")
    #    save_path = saver.save(sess, "./"+ckptname)
    #    print("  Model saved in file: %s" % save_path)

#print("rno=", rno) # (30,20,256)
#print("rns=", rns) # (2,20,256)


# eop
