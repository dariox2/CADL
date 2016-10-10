
#
# testsancho
#
# test saving training checkpoint
# CHECKPOINT DOES NOT WORK, 
# train. saver() does not save what  is 
# supposed to (i.e. cell state variables)
#
# following steps from vaegan example, still doesn't work
#

print("Loading tensorflow...")
import tensorflow as tf
import numpy as np
import os


print("tf version: ", tf.__version__)

print("Reading text file...")

f="../quijote.txt" # only 2000 words approx.

with open(f, 'r') as fp:
    txt = fp.read()

runlimit=1000 # 50~100

vocab = list(set(txt))
print ("txt: ", len(txt), "  vocab: ", len(vocab))

print("Creating encoder...")
encoder = dict(zip(vocab, range(len(vocab))))
print("Creating decoder...")
decoder = dict(zip(range(len(vocab)), vocab))

batch_size = 200  #200

sequence_length = 300 #300

n_cells = 256

n_layers = 2

n_chars = len(vocab)



X = tf.placeholder(tf.int32, [None, sequence_length], name='X')

Y = tf.placeholder(tf.int32, [None, sequence_length], name='Y')


embedding = tf.get_variable("embedding", [n_chars, n_cells])

Xs = tf.nn.embedding_lookup(embedding, X)

print("Xs.get_shape: ", Xs.get_shape().as_list())


with tf.name_scope('reslice'):
    Xs = [tf.squeeze(seq, [1])
          for seq in tf.split(1, sequence_length, Xs)]


cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_cells, state_is_tuple=True)


initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
print("initial state: ", initial_state)

outputs, state = tf.nn.rnn(cells, Xs, initial_state=initial_state)

outputs_flat = tf.reshape(tf.concat(1, outputs), [-1, n_cells])



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

    logits = tf.matmul(outputs_flat, W) + b

    probs = tf.nn.softmax(logits)

    Y_pred = tf.argmax(probs, 1)


print("Creating loss function...")
with tf.variable_scope('loss'):
    Y_true_flat = tf.reshape(tf.concat(1, Y), [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y_true_flat)
    mean_loss = tf.reduce_mean(loss)


print("Creating optimizer & clip gradients...")
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    gradients = []
    clip = tf.constant(5.0, name="clip")
    for grad, var in optimizer.compute_gradients(mean_loss):
        gradients.append((tf.clip_by_value(grad, -clip, clip), var))
    updates = optimizer.apply_gradients(gradients)

sess = tf.Session()

cursor = 0
it_i = 0

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess.run(init)

ckptname="testsancho2_model.ckpt"
print("Restoring model checkpoint...")
if os.path.exists(ckptname):
    saver.restore(sess, ckptname)
    print("  Model restored.")
else:
    print("  Model not found")    

print("Train size: ", batch_size*sequence_length)
print("Begin training...")

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

    loss_val, _ = sess.run([mean_loss, updates],
                           feed_dict={X: Xs, Y: Ys})

    if it_i % 10 == 0:
        print("it_i: ", it_i, "  loss: ", loss_val)
        p = sess.run([Y_pred], feed_dict={X: Xs})[0]
        preds = [decoder[p_i] for p_i in p]
        print("".join(preds).split('\n'))
        print("")

    it_i += 1

    #if it_i % 100 == 0:

saver = tf.train.Saver()
print("Saving checkpoint...")
save_path = saver.save(sess, "./"+ckptname, write_meta_graph=False)
print("  Model saved in file: %s" % save_path)


# eop
