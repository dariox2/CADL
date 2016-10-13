import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from tensorflow.examples.tutorials.mnist import input_data

ds = input_data.read_data_sets('MNIST_data/',one_hot=True)

#dsX = ds.data
#X = load_mnist_images('train-images-idx3-ubyte.gz')
#ds.load_training()

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
    plt.imsave(arr=m, fname='kk_'+saveto)
    return m

print(ds)

print(ds.train.images.shape)

Z = ds.train.images
R = ds.train
i = np.random.randint(0,len(Z))
print(Z.shape)

plt.imshow(ds.train.images[i].reshape((28,28)))
plt.show()

imgs = Z[:1000].reshape((-1,28,28))
plt.imshow(montage(imgs),cmap='gray')
plt.show()

mean_img = np.mean(Z,axis=0)
plt.figure()
plt.imshow(mean_img.reshape((28,28)),cmap='gray')
plt.show()

std_img = np.std(Z,axis=0)
plt.figure()
plt.imshow(std_img.reshape((28,28)))
plt.show()

dimensions = [512,256,128,64]

n_features = Z.shape[1]
print(n_features)

X = tf.placeholder(tf.float32,[None,n_features])
current_input = X
n_input = n_features

Ws = []
for layer_i, n_output in enumerate(dimensions):
	with tf.variable_scope("encoder/layer/{}".format(layer_i)):
		W = tf.get_variable(name='W',shape=[n_input,n_output],initializer=tf.random_normal_initializer(mean=0.0,stddev=0.02))
		h = tf.matmul(current_input,W)
		current_input = tf.nn.relu(h)
		Ws.append(W)
		n_input = n_output

print('Current_input: ',current_input.get_shape())

Ws = Ws[::-1]
print(dimensions[::-1][1:])
dimensions = dimensions[::-1][1:]+[Z.shape[1]]
print(dimensions)

for layer_i, n_output in enumerate(dimensions):
	with tf.variable_scope("decode/layer/{}".format(layer_i)):
		W = tf.transpose(Ws[layer_i])
		h = tf.matmul(current_input,W)
		current_input = tf.nn.relu(h)
		n_input = n_output
		
Y = current_input
print('Y: ',Y)

cost = tf.reduce_mean(tf.squared_difference(X,Y),1)
print(cost.get_shape())
cost = tf.reduce_mean(cost)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_size = 100
n_epochs = 5

examples = Z[:100]

imgs = []
fig, ax = plt.subplots(1,1)
for epoch_i in range(n_epochs):
	for batch_X, _ in ds.train.next_batch():
		sess.run(optimizer,feed_dict={X:batch_X-mean_img})
	recon = sess.run(Y,feed_dict={X:examples - mean_img})
	recon = np.clip((recon + mean_img).reshape((-1,28,28)),0,255)
	img_i = montage(recon).astype(np.unit8)
	imgs.append(img_i)
	ax.imshow(img_i,cmap='gray')
	fig.canvas.draw()
	print(epoch_i,sess.run(cost,feed_dict={X:batch_X - mean_img}))
gif.build_gif(imgs,saveto='kk_ae.gif',cmap='gray')
