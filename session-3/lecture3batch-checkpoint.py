
#
# Lecture 3 - Protobuf
#


import tensorflow as tf
import os


dummy = tf.get_variable(
    name='dummy',
    shape=[2,3],
    initializer=tf.constant_initializer())


sess = tf.Session()
init_op = tf.initialize_all_variables()


saver = tf.train.Saver()
sess.run(init_op)
print("Restoring checkpoint...")
if os.path.exists("model_batch.ckpt"):
    saver.restore(sess, "model_batch.ckpt")
    print("Model restored.")


print("Saving checkpoint...")
save_path = saver.save(sess, "./model_batch.ckpt")
print("Model saved in file: %s" % save_path)

# eop




