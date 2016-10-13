import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.misc import imresize
import pylab
import numpy.random as rnd
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.examples.tutorials.mnist import input_data

def montage(images, saveto='kk_montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.
    Also saves the file to the destination specified by `saveto`.
    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    plt.imsave(arr=m, fname="kk_"+saveto)
    return m

reset_default_graph

ds = input_data.read_data_sets('MNIST_data/',one_hot=True)

n_features = ds.train.images.shape[1]
print('n_features:',n_features)

X = tf.placeholder(tf.float32,[None,n_features])
X_tensor = tf.reshape(X,[-1,28,28,1])

n_filters = [16,16,16]
filter_sizes = [4,4,4]

current_input = X_tensor
n_input = 1
mean_img = np.mean(ds.train.images,axis=0)

Ws = []
shapes = []

print('encode:',n_filters,filter_sizes,shapes)

for layer_i, n_output in enumerate(n_filters):
	with tf.variable_scope("encode/layer/{}".format(layer_i)):
		shapes.append(current_input.get_shape().as_list())
		W = tf.get_variable(
		name='W',
		shape=[filter_sizes[layer_i],
			filter_sizes[layer_i],
			n_input,
			n_output],
		initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02))
		h = tf.nn.conv2d(current_input, W, 
			strides=[1,2,2,1], padding='SAME')
		current_input = tf.nn.relu(h)
		Ws.append(W)
		n_input = n_output
		
Ws.reverse()
shapes.reverse()
n_filters.reverse()
n_filters = n_filters[1:]+[1]

print('reverse:',n_filters,filter_sizes,shapes)

for layer_i,shape in enumerate(shapes):
	with tf.variable_scope("decode/layer/{}".format(layer_i)):
		W = Ws[layer_i]
		h = tf.nn.conv2d_transpose(current_input,W,
			tf.pack([tf.shape(X)[0],shape[1],shape[2],shape[3]]),
			strides=[1,2,2,1],padding='SAME')
		current_input = tf.nn.relu(h)
		
Y = current_input
Y = tf.reshape(Y,[-1,n_features])

cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X,Y),1))
learning_rate = 0.001

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_size = 100
n_epochs = 5
examples = ds.train.images[:100]

imgs = []
fig, ax = plt.subplots(1,1)
for epoch_i in range(n_epochs):
	for batch_X, _ in ds.train.next_batch():
		sess.run(optimizer, feed_dict={X: batch_X - mean_img})
	recon = sess.run(Y,feed_dict={X:examples - mean_img})
	recon = np.clip((recon + mean_img).reshape((-1,28,28)),0,255)
	img_i = montage(recon).astype(np.uint8)
	imgs.append(img_i)
	ax.imshow(img_i, cmap='gray')
	fig.canvas.draw()
	print(epoch_i,sess.run(cost,feed_dict={X: batch_X - mean_img}))
gif.build_gif(imgs, saveto='kk_'+conv-ae.gif',cmap='gray')
	
		





