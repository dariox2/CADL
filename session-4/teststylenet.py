

#
# Style Net
#



#
# VGG Network
#



print("Loading tensorflow...")
import tensorflow as tf
from libs import utils, gif



# dja
import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use('bmh')
import datetime
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
plt.figure(figsize=(5, 5))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from matplotlib.cbook import MatplotlibDeprecationWarning 
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning) 

def wait(n):
    #plt.pause(n)
    plt.pause(1)
    #input("(press enter)")

fncontent="anu455.jpg"
#fnstyle="WP_000478.jpg"
fnstyle="Sharp_Scientific_Calculator_480x800.jpg"
#fncontent=os.path.expanduser("~/fot2.jpg")
#fnstyle="letters-beige.jpg"


# OJO! 500 MB
from libs import vgg16
print("DOWNLOADING VGG16")
net = vgg16.get_vgg_model()



g = tf.Graph()
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    names = [op.name for op in g.get_operations()]



x = g.get_tensor_by_name(names[0] + ':0')
softmax = g.get_tensor_by_name(names[-2] + ':0')


#from skimage.data import coffee
#og = coffee()
og=plt.imread(fncontent)
#plt.imshow(og)
print("IMAGE CONTENT: ", fncontent)
wait(3)

img = vgg16.preprocess(og, dsize=(448,448))

#plt.imshow(vgg16.deprocess(img))
#wait(3)

img_4d = img[np.newaxis]



#
# Dropout
#





#
# Defining the Content Features
#


# DEFINES content_layer!
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    content_layer = 'vgg/conv4_2/conv4_2:0'
    content_features = g.get_tensor_by_name(content_layer).eval(
            session=sess,
            feed_dict={x: img_4d,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]
            })
print("content_features.shape: ", content_features.shape)


#
# Defining the Style Features
#



# Note: Unlike in the lecture, I've cropped the image a bit as the 
# borders took over too much...
style_og = plt.imread(fnstyle)##[15:-15, 190:-190, :]
#plt.title(fnstyle)
#plt.imshow(style_og)
print("IMAGE STYLE: ", fnstyle)
wait(3)


style_img = vgg16.preprocess(style_og, dsize=(448,448))
style_img_4d = style_img[np.newaxis]




style_layers = ['vgg/conv1_1/conv1_1:0',
                'vgg/conv2_1/conv2_1:0',
                'vgg/conv3_1/conv3_1:0',
                'vgg/conv4_1/conv4_1:0',
                'vgg/conv5_1/conv5_1:0']
style_activations = []


with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    for style_i in style_layers:
        style_activation_i = g.get_tensor_by_name(style_i).eval(
            feed_dict={
                x: style_img_4d,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]})
        style_activations.append(style_activation_i)





style_features = []
for style_activation_i in style_activations:
    s_i = np.reshape(style_activation_i, [-1, style_activation_i.shape[-1]])
    gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
    style_features.append(gram_matrix.astype(np.float32))


#
# Remapping the Input
#




tf.reset_default_graph()
g = tf.Graph()




net = vgg16.get_vgg_model()
# net_input = tf.get_variable(
#    name='input',
#    shape=(1, 224, 224, 3),
#    dtype=tf.float32,
#    initializer=tf.random_normal_initializer(
#        mean=np.mean(img), stddev=np.std(img)))
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    net_input = tf.Variable(img_4d)
    tf.import_graph_def(
        net['graph_def'],
        name='vgg',
        input_map={'images:0': net_input})


# Let's take a look at the graph now:

names = [op.name for op in g.get_operations()]
print("vgg graph names: ", names)




#
# Defining the Content Loss
#



with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    content_loss = tf.nn.l2_loss((g.get_tensor_by_name(content_layer)
                   - content_features) / content_features.size)


#
# Defining the Style Loss
#


with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    style_loss = np.float32(0.0)
    for style_layer_i, style_gram_i in zip(style_layers, style_features):
        layer_i = g.get_tensor_by_name(style_layer_i)
        layer_shape = layer_i.get_shape().as_list()
        layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
        layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
        gram_matrix = tf.matmul(tf.transpose(layer_flat), layer_flat) / layer_size
        style_loss = tf.add(style_loss, tf.nn.l2_loss((gram_matrix - style_gram_i) / np.float32(style_gram_i.size)))



#
# Defining the Total Variation Loss
#


def total_variation_loss(x):
    h, w = x.get_shape().as_list()[1], x.get_shape().as_list()[1]
    dx = tf.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
    dy = tf.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
    return tf.reduce_sum(tf.pow(dx + dy, 1.25))

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    tv_loss = total_variation_loss(net_input)



#
# Training
#

# With both content and style losses, we can combine the two, 
# optimizing our loss function, and creating a stylized coffee cup.


with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    loss = 0.1 * content_loss + 5.0 * style_loss + 0.01 * tv_loss
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

print("Training...")
t1 = datetime.datetime.now()
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    sess.run(tf.initialize_all_variables())
    # map input to noise
    n_iterations = 200
    og_img = net_input.eval()
    imgs = []
    for it_i in range(n_iterations):
        _, this_loss, synth = sess.run([optimizer, loss, net_input],
                feed_dict={
                    'vgg/dropout_1/random_uniform:0':
                        np.ones(g.get_tensor_by_name(
                        'vgg/dropout_1/random_uniform:0').get_shape().as_list()),
                    'vgg/dropout/random_uniform:0':
                        np.ones(g.get_tensor_by_name(
                        'vgg/dropout/random_uniform:0').get_shape().as_list())})
        print("It: %d:  loss: %f, (min: %f - max: %f)" %
            (it_i, this_loss, np.min(synth), np.max(synth)))
        if it_i % (n_iterations//50) == 0:
            imgs.append(np.clip(synth[0], 0, 1))
            #fig, ax = plt.subplots(1, 3, figsize=(22, 5))
            #plt.imshow(vgg16.deprocess(img))
            #plt.set_title('content image')
            #plt.imshow(vgg16.deprocess(style_img))
            #plt.set_title('style image')
            plt.title('synthesis #'+str(it_i))
            lastimg=vgg16.deprocess(synth[0])
            plt.imshow(lastimg)
            plt.show()
            #wait(3)
            plt.pause(1)
            # ?fig.canvas.draw()
            plt.imsave(fname='stylenet_last_synth_'+TID+'.png', arr=lastimg)
    gif.build_gif(imgs, saveto='stylenet-test_'+TID+'.gif', interval=200)

t2 = datetime.datetime.now()
delta = t2 - t1
print("             Total animation time: ", delta.total_seconds())
  # Ref: Xubuntu bisal: 408s
  

# eop

