
#
# Lecture 3 - Protobuf
#


import tensorflow as tf
import os

# 1) Checkpoint

dummy = tf.get_variable(
    name='dummy',
    shape=[2,3],
    initializer=tf.constant_initializer())


sess = tf.Session()
init_op = tf.initialize_all_variables()


saver = tf.train.Saver()
sess.run(init_op)
print("Restoring checkpoint...")
if os.path.exists("model_batch_protobuf.ckpt"):
    saver.restore(sess, "model_batch_protobuf.ckpt")
    print("Model restored.")


print("Saving checkpoint...")
save_path = saver.save(sess, "./model_batch_protobuf.ckpt")
print("Model saved in file: %s" % save_path)

# 2) Protobuf

path='./'
ckpt_name = 'model.ckpt'
fname = 'model.tfmodel'
dst_nodes = ['Y']
g_1 = tf.Graph()
with tf.Session(graph=g_1) as sess:
    x = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    # Replace this with some code which will create your tensorflow graph:
    net = create_network()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, ckpt_name)
    graph_def = tf.python.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, dst_nodes)
g_2 = tf.Graph()
with tf.Session(graph=g_2) as sess:
    tf.train.write_graph(
        tf.python.graph_util.extract_sub_graph(
            graph_def, dst_nodes), path, fname, as_text=False)

with open("model.tfmodel", mode='rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

tf.import_graph_def(net['graph_def'], name='model')



# eop




