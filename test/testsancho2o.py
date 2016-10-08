
# testsancho2s - simple
# CHECKPOINT DOES NOT WORK, 

print("Loading tensorflow...")
import tensorflow as tf
import numpy as np
import os


ckptname="tmp/testsancho2_model.ckpt"
runlimit=500 #50  # 50~100
mustsave=False
mustrestore=True

with tf.Session() as sess:

  print("Reading text file...")

  f="bohemian.txt" # only 2000 words approx.
  with open(f, 'r') as fp:
    txt = fp.read()

  vocab = list(set(txt))
  print ("txt: ", len(txt), "  vocab: ", len(vocab))

  encoder = dict(zip(vocab, range(len(vocab))))
  decoder = dict(zip(range(len(vocab)), vocab))

  batch_size = 20
  sequence_length = 30
  n_cells = 256
  n_layers = 2
  n_chars = len(vocab)

  X = tf.placeholder(tf.int32, [None, sequence_length], name='X')
  Y = tf.placeholder(tf.int32, [None, sequence_length], name='Y')

  embedding = tf.get_variable("embedding", [n_chars, n_cells])
  Xs = tf.nn.embedding_lookup(embedding, X)
  print("Xs.get_shape: ", Xs.get_shape().as_list())

  #with tf.name_scope('reslice'):
  Xs = [tf.squeeze(seq, [1])
         for seq in tf.split(1, sequence_length, Xs)]

  cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_cells,
                   state_is_tuple=True)

  #
  # 1) OK FOR INITIAL TRAINING; STATE IS LOST AFTER
  #
  #initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
  #print("initial state: ", initial_state)
  #outputs, state = tf.nn.rnn(cells, Xs, initial_state=initial_state)
  
  #
  # 2) OK(????) FOR RESTORING AND EVALUATE OUTPUT; DEGRADES IF TRAINED
  #
  outputs, state = tf.nn.rnn(cells, Xs, dtype=tf.float32)


  outputs_flat = tf.reshape(tf.concat(1, outputs), [-1, n_cells])

  print("Creating prediction layer...")
  #with tf.variable_scope('prediction'):
  #W = tf.get_variable(        "W",        shape=[n_cells, n_chars])
  #b = tf.get_variable(        "b",        shape=[n_chars])

  #with tf.variable_scope('loss'):
  Y_true_flat = tf.reshape(tf.concat(1, Y), [-1])

  #with tf.name_scope('optimizer'):
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  gradients = []
  clip = tf.constant(5.0, name="clip")

  cursor = 0
  it_i = 0

  print("  Initializing...")    
  init = tf.initialize_all_variables()
  sess.run(init)

  print("ANTES=============")
  for tensor in tf.get_default_graph().as_graph_def().node:
    print(tensor)

  saver = tf.train.Saver()

  if mustrestore and os.path.exists(ckptname):
    saver.restore(sess, ckptname)
    print("  Model restored.")

  print("DESPUES=============")
  for tensor in tf.get_default_graph().as_graph_def().node:
    print(tensor)


  W=tf.get_default_graph().get_tensor_by_name("W")
  b=tf.get_default_graph().get_tensor_by_name("b:0")

  logits = tf.matmul(outputs_flat, W) + b
  probs = tf.nn.softmax(logits)
  Y_pred = tf.argmax(probs, 1)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
            Y_true_flat)
  mean_loss = tf.reduce_mean(loss)
  for grad, var in optimizer.compute_gradients(mean_loss):
      gradients.append((tf.clip_by_value(grad, -clip, clip), var))
  updates = optimizer.apply_gradients(gradients)

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
    #loss_val = sess.run([mean_loss],
                           feed_dict={X: Xs, Y: Ys})

    if it_i % 10 == 0:
      print("it_i: ", it_i, "  loss: ", loss_val)
      p = sess.run([Y_pred], feed_dict={X: Xs})[0]
      preds = [decoder[p_i] for p_i in p]
      print("".join(preds).split('\n'))
      print("")

    it_i += 1

  if mustsave:
    print("Saving checkpoint...")
    save_path = saver.save(sess, ckptname)
    print("  Model saved in file: %s" % save_path)

# eop

