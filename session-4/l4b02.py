
#
# Session 4 part 3: Style Net
#

#
# Style Net
#

# Leon Gatys and his co-authors demonstrated a pretty epic extension 
# to deep dream which showed that neural networks trained on objects 
# like the one we've been using actually represent both content and 
# style, and that these can be independently manipulated, for 
# instance taking the content from one image, and the style from 
# another. They showed how you could artistically stylize the same 
# image with a wide range of different painterly aesthetics. Let's 
# take a look at how we can do that. We're going to use the same 
# network that they've used in their paper, VGG. This network is a 
# lot less complicated than the Inception network, but at the 
# expense of having a lot more parameters.

#
# VGG Network
#

# In the resources section, you can find the library for loading 
# this network, just like you've done w/ the Inception network. 
# Let's reset the graph:

print("Loading tensorflow...")
import tensorflow as tf
from libs import utils

#from tensorflow.python.framework.ops import reset_default_graph
#sess.close()
#reset_default_graph()

# dja
#import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
import datetime
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
plt.figure(figsize=(5, 5))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")


# And now we'll load up the new network, except unlike before, we're 
# going to explicitly create a graph, and tell the session to use 
# this graph. If we didn't do this, tensorflow would just use the 
# default graph that is always there. But since we're going to be 
# making a few graphs, we'll need to do it like this.

# OJO! 500 MB
from libs import vgg16
print("DOWNLOADING VGG16")
net = vgg16.get_vgg_model()

# Note: We will explicitly define a context manager here to handle 
# the graph and place the graph in CPU memory instead of GPU memory, 
# as this is a very large network!

g = tf.Graph()
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    names = [op.name for op in g.get_operations()]

# Let's take a look at the network:

# REQUIRES TENSORBOARD
# nb_utils.show_graph(net['graph_def'])

print("names: ", names)



So unlike inception, which has many parallel streams and concatenation operations, this network is much like the network we've created in the last session. A pretty basic deep convolutional network with a single stream of many convolutions, followed by adding biases, and using relu non-linearities.

Let's grab a placeholder for the input and output of the network:

x = g.get_tensor_by_name(names[0] + ':0')
softmax = g.get_tensor_by_name(names[-2] + ':0')

We'll grab an image preprocess, add a new dimension to make the image 4-D, then predict the label of this image just like we did with the Inception network:

from skimage.data import coffee
og = coffee()
plt.imshow(og)

img = vgg16.preprocess(og)

plt.imshow(vgg16.deprocess(img))

img_4d = img[np.newaxis]

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={x: img_4d})[0]
    print([(res[idx], net['labels'][idx])
           for idx in res.argsort()[-5:][::-1]])


#
# Dropout
#

If I run this again, I get a different result:


with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={x: img_4d})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])

That's because this network is using something called dropout. Basically, dropout will randomly drop connections. This is useful because it allows multiple paths of explanations for a network. Consider how this might be manifested in an image recognition network. Perhaps part of the object is occluded. We would still want the network to be able to describe the object. That's a very useful thing to do during training to do what's called regularization. Basically regularization is a fancy term for make sure the activations are within a certain range which I won't get into there. It turns out there are other very good ways of performing regularization including dropping entire layers instead of indvidual neurons; or performing what's called batch normalization, which I also won't get into here.

To use the VGG network without dropout, we'll have to set the values of the dropout "keep" probability to be 1, meaning don't drop any connections:

[name_i for name_i in names if 'dropout' in name_i]

Looking at the network, it looks like there are 2 dropout layers. Let's set these values to 1 by telling the feed_dict parameter.


with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={
        x: img_4d,
        'vgg/dropout_1/random_uniform:0': [[1.0]],
        'vgg/dropout/random_uniform:0': [[1.0]]})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])

Let's try again to be sure:

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={
        x: img_4d,
        'vgg/dropout_1/random_uniform:0': [[1.0]],
        'vgg/dropout/random_uniform:0': [[1.0]]})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])

Great so we get the exact same probability and it works just like the Inception network!


#
# Defining the Content Features
#

For the "content" of the image, we're going to need to know what's happening in the image at the broadest spatial scale. Remember before when we talked about deeper layers having a wider receptive field? We're going to use that knowledge to say that the later layers are better at representing the overall content of the image. Let's try using the 4th layer's convolution for the determining the content:

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    content_layer = 'vgg/conv4_2/conv4_2:0'
    content_features = g.get_tensor_by_name(content_layer).eval(
            session=sess,
            feed_dict={x: img_4d,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]
            })
print(content_features.shape)


#
# Defining the Style Features
#

Great. We now have a tensor describing the content of our original image. We're going to stylize it now using another image. We'll need to grab another image. I'm going to use Hieronymous Boschs's famous still life painting of sunflowers.

filepath = utils.download('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/El_jard%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg/640px-El_jard%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg')

# Note: Unlike in the lecture, I've cropped the image a bit as the borders took over too much...
style_og = plt.imread(filepath)[15:-15, 190:-190, :]
plt.imshow(style_og)

We'll need to preprocess it just like we've done with the image of the espresso:

style_img = vgg16.preprocess(style_og)
style_img_4d = style_img[np.newaxis]

And for fun let's see what VGG thinks of it:

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(
        feed_dict={
            x: style_img_4d,
            'vgg/dropout_1/random_uniform:0': [[1.0]],
            'vgg/dropout/random_uniform:0': [[1.0]]})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])


So it's not great. It looks like it thinks it's a jigsaw puzzle. What we're going to do is find features of this image at different layers in the network.


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


Instead of using the raw activations of these layers, what the authors of the StyleNet paper suggest is to use the Gram activation of the layers instead, which mathematically is expressed as the matrix transpose multiplied by itself. The intuition behind this process is that it measures the similarity between every feature of a matrix. Or put another way, it is saying how often certain features appear together.

This would seem useful for "style", as what we're trying to do is see what's similar across the image. To get every feature, we're going to have to reshape our N x H x W x C matrix to have every pixel belonging to each feature in a single column. This way, when we take the transpose and multiply it against itself, we're measuring the shared direction of every feature with every other feature. Intuitively, this would be useful as a measure of style, since we're measuring whats in common across all pixels and features.


style_features = []
for style_activation_i in style_activations:
    s_i = np.reshape(style_activation_i, [-1, style_activation_i.shape[-1]])
    gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
    style_features.append(gram_matrix.astype(np.float32))



#
# Remapping the Input
#

So now we have a collection of "features", which are basically the activations of our sunflower image at different layers. We're now going to try and make our coffee image have the same style as this image by trying to enforce these features on the image. Let's take a look at how we can do that.

We're going to need to create a new graph which replaces the input of the original VGG network with a variable which can be optimized. So instead of having a placeholder as input to the network, we're going to tell tensorflow that we want this to be a tf.Variable. That's because we're going to try to optimize what this is, based on the objectives which we'll soon create.


reset_default_graph()
g = tf.Graph()


And now we'll load up the VGG network again, except unlike before, we're going to map the input of this network to a new variable randomly initialized to our content image. Alternatively, we could initialize this image noise to see a different result.


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


Let's take a look at the graph now:


names = [op.name for op in g.get_operations()]
print(names)


So notice now the first layers of the network have everything prefixed by input, our new variable which we've just created. This will initialize a variable with the content image upon initialization. And then as we run whatever our optimizer ends up being, it will slowly become the a stylized image.

#
# Defining the Content Loss
#


We now need to define a loss function which tries to optimize the distance between the net's output at our content layer, and the content features which we have built from the coffee image:



with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    content_loss = tf.nn.l2_loss((g.get_tensor_by_name(content_layer) -
                                 content_features) /
                                 content_features.size)


#
# Defining the Style Loss
#

For our style loss, we'll compute the gram matrix of the current network output, and then measure the l2 loss with our precomputed style image's gram matrix. So most of this is the same as when we compute the gram matrix for the style image, except now, we're doing this in tensorflow's computational graph, so that we can later connect these operations to an optimizer. Refer to the lecture for a more in depth explanation of this.


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

Lastly, we'll create a third loss value which will simply measure the difference between neighboring pixels. By including this as a loss, we're saying that we want neighboring pixels to be similar.



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

With both content and style losses, we can combine the two, optimizing our loss function, and creating a stylized coffee cup.


with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    loss = 0.1 * content_loss + 5.0 * style_loss + 0.01 * tv_loss
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)


with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    sess.run(tf.initialize_all_variables())
    # map input to noise
    n_iterations = 100
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
        print("%d: %f, (%f - %f)" %
            (it_i, this_loss, np.min(synth), np.max(synth)))
        if it_i % 5 == 0:
            imgs.append(np.clip(synth[0], 0, 1))
            fig, ax = plt.subplots(1, 3, figsize=(22, 5))
            ax[0].imshow(vgg16.deprocess(img))
            ax[0].set_title('content image')
            ax[1].imshow(vgg16.deprocess(style_img))
            ax[1].set_title('style image')
            ax[2].set_title('current synthesis')
            ax[2].imshow(vgg16.deprocess(synth[0]))
            plt.show()
            fig.canvas.draw()
    gif.build_gif(imgs, saveto='stylenet-bosch.gif')


We can play with a lot of the parameters involved to produce wildly different results. There are also a lot of extensions to what I've presented here currently in the literature including incorporating structure, temporal constraints, variational constraints, and other regularizing methods including making use of the activations in the content image to help infer what features in the gram matrix are relevant.

There is also no reason I can see why this approach wouldn't work with using different sets of layers or different networks entirely such as the Inception network we started with in this session. Perhaps after exploring deep representations a bit more, you might find intuition towards which networks, layers, or neurons in particular represent the aspects of the style you want to bring out. You might even try blending different sets of neurons to produce interesting results. Play with different motions. Try blending the results as you produce the deep dream with other content.

Also, there is no reason you have to start with an image of noise, or an image of the content. Perhaps you can start with an entirely different image which tries to reflect the process you are interested in. There are also a lot of interesting published extensions to this technique including image analogies, neural doodle, incorporating structure, and incorporating temporal losses from optical flow to stylize video.

There is certainly a lot of room to explore within technique. A good starting place for the possibilities with the basic version of style net I've shown here is Kyle McDonald's Style Studies:

http://www.kylemcdonald.net/stylestudies/

If you find other interesting applications of the technique, feel free to post them on the forums.





















input("End")

# eop

