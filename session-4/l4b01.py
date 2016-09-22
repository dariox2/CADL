
#
# Session 4 - Visualizing Representations

print("Begin...")

#
# Introduction
#

# So far, we've seen that a deep convolutional network can get very 
# high accuracy in classifying the MNIST dataset, a dataset of 
# handwritten digits numbered 0 - 9. What happens when the number 
# of classes grows higher than 10 possibilities? Or the images get
# much larger? We're going to explore a few new datasets and bigger
# and better models to try and find out. We'll then explore a few 
# interesting visualization tehcniques to help us understand what
# the networks are representing in its deeper layers and how these 
# techniques can be used for some very interesting creative 
# applications.

#
# Deep Convolutional Networks
#

# Almost 30 years of computer vision and machine learning research 
# based on images takes an approach to processing images like what 
# we saw at the end of Session 1: you take an image, convolve it
# with a set of edge detectors like the gabor filter we created, 
# and then find some thresholding of this image to find more 
# interesting features, such as corners, or look at histograms 
# of the number of some orientation of edges in a particular window.
# In the previous session, we started to see how Deep Learning has 
# allowed us to move away from hand crafted features such as 
# Gabor-like filters to letting data discover representations. 
# Though, how well does it scale?
# 
# A seminal shift in the perceived capabilities of deep neural 
# networks occurred in 2012. A network dubbed AlexNet, after its 
# primary author, Alex Krizevsky, achieved remarkable performance
# on one of the most difficult computer vision datasets at the time,
# ImageNet. . ImageNet is a dataset used in a yearly challenge 
# called the ImageNet Large Scale Visual Recognition Challenge 
# (ILSVRC), started in 2010. The dataset contains nearly 1.2 million
# images composed of 1000 different types of objects. Each object
# has anywhere between 600 - 1200 different images.
#
# Up until now, the most number of labels we've considered is 10!
# The image sizes were also very small, only 28 x 28 pixels, and it
# didn't even have color.
#
# Let's look at a state-of-the-art network that has already been 
# trained on ImageNet.

#
# Loading a Pretrained Network
#

# We can use an existing network that has been trained by loading 
# the model's weights into a network definition. The network
# definition is basically saying what are the set of operations
# in the tensorflow graph. So how is the image manipulated, 
# filtered, in order to get from an input image to a probability
# saying which 1 of 1000 possible objects is the image describing? 
# It also restores the model's weights. Those are the values of 
# every parameter in the network learned through gradient descent.
# Luckily, many researchers are releasing their model definitions 
# and weights so we don't have to train them! We just have to load
# them up and then we can use the model straight away. That's very
# lucky for us because these models take a lot of time, cpu, memory,
# and money to train.
# 
# To get the files required for these models, you'll need to 
# download them from the resources page.
#
# First, let's import some necessary libraries.

import numpy as np
import matplotlib.pyplot as plt

print("Loading tensorflow...")
import tensorflow as tf

#import IPython.display as ipyd
from libs import gif, nb_utils


# dja
plt.style.use('bmh')
import datetime
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
plt.figure(figsize=(5, 5))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")

# Start an interactive session:
sess = tf.InteractiveSession()

# Now we'll load Google's Inception model, which is a pretrained
# network for classification built using the ImageNet database. 
# I've included some helper functions for getting this model
# loaded and setup w/ Tensorflow.
from libs import inception
print("Loading inception...")
net = inception.get_inception_model()

# Here's a little extra that wasn't in the lecture. We can visualize
# the graph definition using the nb_utils module's show_graph
# function. This function is taken from an example in the Tensorflow
# repo so I can't take credit for it! It uses Tensorboard, which we
# didn't get a chance to discuss, Tensorflow's web interface for
# visualizing graphs and training performance. It is very useful but
# we sadly did not have enough time to discuss this!

# REQUIERE TENSORBOARD
#nb_utils.show_graph(net['graph_def'])

# We'll now get the graph from the storage container, and tell 
# tensorflow to use this as its own graph. This will add all the
# computations we need to compute the entire deep net, as well as
# all of the pre-trained parameters.

tf.import_graph_def(net['graph_def'], name='inception')

print("net labels: ", net['labels'])

# Let's have a look at the graph:

g = tf.get_default_graph()
names = [op.name for op in g.get_operations()]
print("names: ", names)

# The input to the graph is stored in the first tensor output, and
# the probability of the 1000 possible objects is in the last layer:
input_name = names[0] + ':0'
x = g.get_tensor_by_name(input_name)

softmax = g.get_tensor_by_name(names[-1] + ':0')


# Predicting with the Inception Network
# Let's try to use the network to predict now:
from skimage.data import coffee
og = coffee()
plt.title("original coffee")
plt.imshow(og)
plt.show()
plt.pause(3)
print("og min, max: ", og.min(), og.max())

# We'll crop and resize the image to 224 x 224 pixels. I've provided
# a simple helper function which will do this for us

# Note that in the lecture, I used a slightly different inception
# model, and this one requires us to subtract the mean from the input
# image. The preprocess function will also crop/resize the image to
# 299x299
img = inception.preprocess(og)
print("og.shape: ", og.shape), print("img.shapw: ", img.shape)

# So this will now be a different range than what we had in the lecture:
print("img min,max: ", img.min(), img.max())

# As we've seen from the last session, our images must be shaped as a
# 4-dimensional shape describing the number of images, height, width, and
# number of channels. So our original 3-dimensional image of height,
# width, channels needs an additional dimension on the 0th axis.

img_4d = img[np.newaxis]
print("img_4d.shape: ", img_4d.shape)


# Note that unlike the lecture, we have to call the `inception.deprocess`
# function so that it adds back the mean!
plt.title("deprocess")
plt.imshow(inception.deprocess(img))
plt.show()
plt.pause(3)


res = np.squeeze(softmax.eval(feed_dict={x: img_4d}))

# Note that this network is slightly different than the one used in the
# lecture. Instead of just 1 output, there will be 16 outputs of 1008
# probabilities. We only use the first 1000 probabilities (the extra ones
# are for negative/unseen labels)
print("result shape: ", res.shape)

# The result of the network is a 1000 element vector, with probabilities
# of each class. Inside our net dictionary are the labels for every
# element. We can sort these and use the labels of the 1000 classes to
# see what the top 5 predicted probabilities and labels are:

# Note that this is one way to aggregate the different probabilities.  We # could also take the argmax.
res = np.mean(res, 0)
res = res / np.sum(res)

print("result idx,label: ")
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])

# Visualizing Filters
#
# Wow so it works! But how!? Well that's an ongoing research question.
# There has been a lot of great developments in the last few years to
# help us understand what might be happening. Let's try to first
# visualize the weights of the convolution filters, like we've done with
# our MNIST network before.

W = g.get_tensor_by_name('inception/conv2d0_w:0')
W_eval = W.eval()
print("W_eval.shape: ", W_eval.shape)

# With MNIST, our input number of filters was 1, since our input number
# of channels was also 1, as all of MNIST is grayscale. But in this case,
# our input number of channels is 3, and so the input number of
# convolution filters is also 3. We can try to see every single
# individual filter using the library tool I've provided:

from libs import utils
W_montage = utils.montage_filters(W_eval)
plt.title("filters montage")
plt.imshow(W_montage, interpolation='nearest')
plt.show()
plt.pause(3)

# Or, we can also try to look at them as RGB filters, showing the
# influence of each color channel, for each neuron or output filter.

Ws = [utils.montage_filters(W_eval[:, :, [i], :]) for i in range(3)]
Ws = np.rollaxis(np.array(Ws), 0, 3)
plt.title("as rgb filters")
plt.imshow(Ws, interpolation='nearest')
plt.show()
plt.pause(5)

# In order to better see what these are doing, let's normalize the
# filters range:

np.min(Ws), np.max(Ws)
Ws = (Ws / np.max(np.abs(Ws)) * 128 + 128).astype(np.uint8)
plt.title("normalized filters")
plt.imshow(Ws, interpolation='nearest')
plt.show()
plt.pause(3)

# Like with our MNIST example, we can probably guess what some of these
# are doing. They are responding to edges, corners, and center-surround
# or some kind of contrast of two things, like red, green, blue yellow,
# which interestingly is also what neuroscience of vision tells us about
# how the human vision identifies color, which is through opponency of
# red/green and blue/yellow. To get a better sense, we can try to look at
# the output of the convolution:

feature = g.get_tensor_by_name('inception/conv2d0_pre_relu:0')

# Let's look at the shape:

layer_shape = tf.shape(feature).eval(feed_dict={x:img_4d})
print("layer_shape: ", layer_shape)

# So our original image which was 1 x 224 x 224 x 3 color channels, now
# has 64 new channels of information. The image's height and width are
# also halved, because of the stride of 2 in the convolution. We've just
# seen what each of the convolution filters look like. Let's try  to see
# how they filter the image now by looking at the resulting convolution.

f = feature.eval(feed_dict={x: img_4d})
montage = utils.montage_filters(np.rollaxis(np.expand_dims(f[0], 3), 3, 2))
plt.title('deprocess orig')
plt.imshow(inception.deprocess(img))
plt.show()
plt.pause(3)

plt.title('Convolution Filters')
plt.imshow(Ws, interpolation='nearest')
plt.show()
plt.pause(3)

plt.title('Convolution Outputs')
plt.imshow(montage, cmap='gray')
plt.show()
plt.pause(3)

# it's a little hard to see what's happening here but let's try. The
# third filter for instance seems to be a lot like the gabor filter we
# created in the first session. It respond to horizontal edges, since it
# has a bright component at the top, and a dark component on the bottom.
# Looking at the output of the convolution, we can see that the
# horizontal edges really pop out.

#
#Visualizing the Gradient
#

# So this is a pretty useful technique for the first convolution layer.
# But when we get to the next layer, all of sudden we have 64 different
# channels of information being fed to more convolution filters of some
# very high dimensions. It's very hard to conceptualize that many
# dimensions, let alone also try and figure out what it could be doing
# with all the possible combinations it has with other neurons in other
# layers.
#
# If we want to understand what the deeper layers are really doing, we're
# going to have to start to use backprop to show us the gradients of a
# particular neuron with respect to our input image. Let's visualize the
# network's gradient activation when backpropagated to the original input
# image. This is effectively telling us which pixels are responding to the
# predicted class or given neuron.
#
# We use a forward pass up to the layer that we are interested in, and
# then a backprop to help us understand what pixels in particular
# contributed to the final activation of that layer. We will need to
# create an operation which will find the max neuron of all activations
# in a layer, and then calculate the gradient of that objective with
# respect to the input image.

feature = g.get_tensor_by_name('inception/conv2d0_pre_relu:0')
gradient = tf.gradients(tf.reduce_max(feature, 3), x)

# When we run this network now, we will specify the gradient operation
# we've created, instead of the softmax layer of the network. This will
# run a forward prop up to the layer we asked to find the gradient with,
# and then run a back prop all the way to the input image.

print("Running prop up, finding gradient...")
res = sess.run(gradient, feed_dict={x: img_4d})[0]

# Let's visualize the original image and the output of the backpropagated
# gradient:

plt.title("inception deprocess")
plt.imshow(inception.deprocess(img))
plt.show()
plt.pause(3)

plt.title("result 0")
plt.imshow(res[0])
plt.show()
plt.pause(3)

# Well that looks like a complete mess! What we can do is normalize the
# activations in a way that let's us see it more in terms of the normal
# range of color values.

def normalize(img, s=0.1):
    '''Normalize the image range for visualization'''
    z = img / np.std(img)
    return np.uint8(np.clip(
        (z - z.mean()) / max(z.std(), 1e-4) * s + 0.5,
        0, 1) * 255)


r = normalize(res)
plt.title("normalized deprocess")
plt.imshow(inception.deprocess(img))
plt.show()
plt.pause(3)

plt.title("normalized result")
plt.imshow(r[0])
plt.show()
plt.pause(3)

# Much better! This sort of makes sense! There are some strong edges and
# we can really see what colors are changing along those edges.
#
# We can try within individual layers as well, pulling out individual
# neurons to see what each of them are responding to. Let's first create
# a few functions which will help us visualize a single neuron in a
# layer, and every neuron of a layer:

def compute_gradient(input_placeholder, img, layer_name, neuron_i):
    feature = g.get_tensor_by_name(layer_name)
    gradient = tf.gradients(tf.reduce_mean(feature[:, :, :, neuron_i]), x)
    res = sess.run(gradient, feed_dict={input_placeholder: img})[0]
    return res


def compute_gradients(input_placeholder, img, layer_name):
    feature = g.get_tensor_by_name(layer_name)
    layer_shape = tf.shape(feature).eval(feed_dict={input_placeholder: img})
    gradients = []
    for neuron_i in range(layer_shape[-1]):
        gradients.append(compute_gradient(input_placeholder, img, layer_name, neuron_i))
    return gradients

# Now we can pass in a layer name, and see the gradient of every neuron
# in that layer with respect to the input image as a montage. Let's try
# the second convolutional layer. This can take awhile depending on your
# computer:

gradients = compute_gradients(x, img_4d, 'inception/conv2d1_pre_relu:0')
gradients_norm = [normalize(gradient_i[0]) for gradient_i in gradients]
montage = utils.montage(np.array(gradients_norm))

plt.title("gradients norm montage")
plt.imshow(montage)
plt.show()
plt.pause(3)

# So it's clear that each neuron is responding to some type of feature.
# It looks like a lot of them are interested in the texture of the cup,
# and seem to respond in different ways across the image. Some seem to be
# more interested in the shape of the cup, responding pretty strongly to
# the circular opening, while others seem to catch the liquid in the cup
# more. There even seems to be one that just responds to the spoon, and
# another which responds to only the plate.
#
# Let's try to get a sense of how the activations in each layer progress.
# We can get every max pooling layer like so:

features = [name for name in names if 'maxpool' in name.split()[-1]]
print("features: ", features)

# So I didn't mention what max pooling is. But it 's a simple operation.
# You can think of it like a convolution, except instead of using a
# learned kernel, it will just find the maximum value in the window, for
# performing "max pooling", or find the average value, for performing 
# "average pooling".
#
# We'll now loop over every feature and create an operation that first
# will find the maximally activated neuron. It will then find the sum of
# all activations across every pixel and input channel of this neuron,
# and then calculate its gradient with respect to the input image.

n_plots = len(features) + 1
base = img_4d
plt.title("feature loop")
plt.imshow(inception.deprocess(img))
plt.show()
plt.pause(3)
for feature_i, featurename in enumerate(features):
    feature = g.get_tensor_by_name(featurename + ':0')
    neuron = tf.reduce_max(feature, len(feature.get_shape())-1)
    gradient = tf.gradients(tf.reduce_sum(neuron), x)
    this_res = sess.run(gradient[0], feed_dict={x: base})[0]
    plt.title("feature: "+featurename)
    plt.imshow(normalize(this_res))
    plt.show()
    plt.pause(3)

# To really understand what's happening in these later layers, we're
# going to have to experiment with some other visualization techniques.


###################################################################
###################################################################

# Deep Dreaming
#
# Sometime in May of 2015, A researcher at Google, Alexander Mordvintsev,
# took a deep network meant to recognize objects in an image, and instead
# used it to *generate new objects in an image. The internet quickly
# exploded after seeing one of the images it produced. Soon after, Google
# posted a blog entry on how to perform the technique they re-dubbed
# "Inceptionism", and tons of interesting outputs were soon created.
# Somehow the name Deep Dreaming caught on, and tons of new creative
# applications came out, from twitter bots (DeepForger), to streaming
# television (twitch.tv), to apps, it was soon everywhere.
#
# What Deep Dreaming is doing is taking the backpropagated gradient
# activations and simply adding it back to the image, running the same
# process again and again in a loop. I think "dreaming" is a great
# description of what's going on. We're really pushing the network in a
# direction, and seeing what happens when left to its devices. What it is
# effectively doing is amplifying whatever our objective is, but we get
# to see how that objective is optimized in the input space rather than
# deep in the network in some arbitrarily high dimensional space that no
# one can understand.
#
# There are many tricks one can add to this idea, such as blurring,
# adding constraints on the total activations, decaying the gradient,
# infinitely zooming into the image by cropping and scaling, adding
# jitter by randomly moving the image around, or plenty of other ideas
# waiting to be explored.
#
# Simplest Approach
#
# Let's try the simplest approach for deep dream using a few of these
# layers. We're going to try the first max pooling layer to begin with.
# We'll specify our objective which is to follow the gradient of the mean
# of the selected layers's activation. What we should see is that same
# objective being amplified so that we can start to understand in terms
# of the input image what the mean activation of that layer tends to
# like, or respond to. We'll also produce a gif of every few frames. For
# the remainder of this section, we'll need to rescale our 0-255 range
# image to 0-1 as it will speed up things:

print("DEEP DREAMING")

# Rescale to 0-1 range
img_4d = img_4d / np.max(img_4d)

# Get the max pool layer
layer = g.get_tensor_by_name('inception/maxpool0:0')

# Find the gradient of this layer's mean activation with respect to the input image
gradient = tf.gradients(tf.reduce_mean(layer), x)

# Copy the input image as we'll add the gradient to it in a loop
img_copy = img_4d.copy()

# We'll run it for 50 iterations
n_iterations = 50

# Think of this as our learning rate.  This is how much of the gradient we'll add to the input image
step = 1.0

# Every 10 iterations, we'll add an image to a GIF
gif_step = 10

# Storage for our GIF
imgs = []
print("deep dreaming:")
for it_i in range(n_iterations):

    print(it_i, end=', ')

    # This will calculate the gradient of the layer we chose with respect to the input image.
    this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]

    # Let's normalize it by the maximum activation
    this_res /= (np.max(np.abs(this_res)) + 1e-8)

    # Then add it to the input image
    img_copy += this_res * step

    # And add to our gif
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))

print("")

# Build the gif
gif.build_gif(imgs, saveto='1-simplest-mean-layer_' +
 TID + '.gif', interval=0.3, show_gif=False)


# What we can see is pretty quickly, the activations tends to pick up the
# fine detailed edges of the cup, plate, and spoon. Their structure is
# very local, meaning they are really describing information at a very
# small scale.
#
# We could also specify the maximal neuron's mean activation, instead of
# the mean of the entire layer:
#

# Find the maximal neuron in a layer
neuron = tf.reduce_max(layer, len(layer.get_shape())-1)
# Then find the mean over this neuron
gradient = tf.gradients(tf.reduce_mean(neuron), x)

# The rest is exactly the same as before:

img_copy = img_4d.copy()
imgs = []
print("gif max neuron:")
for it_i in range(n_iterations):
    print(it_i,  end=', ')
    this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
print("")

gif.build_gif(imgs, saveto='1-simplest-max-neuron_' + TID + '.gif',
   interval=0.3, show_gif=False)

# What we should see here is how the maximal neuron in a layer's
# activation is slowly maximized through gradient ascent. So over time,
# we're increasing the overall activation of the neuron we asked for.
#
# Let's try doing this for each of our max pool layers, in increasing
# depth, and let it run a little longer. This will take a long time
# depending on your machine!

# For each max pooling feature, we'll produce a GIF
for feature_i in features:
    print("gif feature: ", feature_i)
    layer = g.get_tensor_by_name(feature_i + ':0')
    gradient = tf.gradients(tf.reduce_mean(layer), x)
    img_copy = img_4d.copy()
    imgs = []
    for it_i in range(n_iterations):
        print(it_i, end=', ')
        this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]
        this_res /= (np.max(np.abs(this_res)) + 1e-8)
        img_copy += this_res * step
        if it_i % gif_step == 0:
            imgs.append(normalize(img_copy[0]))
    print(" build gif")
    gif.build_gif(imgs, saveto='1-simplest-' + 
        feature_i.split('/')[-1] + '_' + TID + '.gif',
         interval=0.3, show_gif=False)
    plt.pause(1)
# When we look at the outputs of these, we should see the representations
# in corresponding layers being amplified on the original input image.
# As we get to later layers, it really starts to appear to hallucinate,
# and the patterns start to get more complex. That's not all though. 
# The patterns also seem to grow larger. What that means is that at 
# later layers, the representations span a larger part of the image.
# In neuroscience, we might say that this has a larger receptive field,
# since it is receptive to the content in a wider visual field.
#
# Let's try the same thing except now we'll feed in noise instead of
# an image:

# Create some noise, centered at gray
img_noise = inception.preprocess(
    (np.random.randint(100, 150, size=(224, 224, 3))))[np.newaxis]
print("img noise min/max: ", img_noise.min(), img_noise.max())
# ERROR
#  File "l4b01.py", line 612, in <module>
#    (np.random.randint(100, 150, size=(224, 224, 3))))[np.newaxis]
#  File "/home/dario/cadl/session-4/libs/inception.py", line 83, in # preprocess
#    img *= 255.0
# TypeError: Cannot cast ufunc multiply output from dtype('float64') 
# to dtype('int64') with casting rule 'same_kind'


# And the rest is the same:

print("Feeding noise:")
for feature_i in features:
    print("feature: ", feature_i)
    layer = g.get_tensor_by_name(feature_i + ':0')
    gradient = tf.gradients(tf.reduce_mean(layer), x)
    img_copy = img_noise.copy()
    imgs = []
    for it_i in range(n_iterations):
        print(it_i, end=', ')
        this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]
        this_res /= (np.max(np.abs(this_res)) + 1e-8)
        img_copy += this_res * step
        if it_i % gif_step == 0:
            imgs.append(normalize(img_copy[0]))
    print(" build_gif")
    gif.build_gif(
        imgs, saveto='1-simplest-noise-' + feature_i.split('/')[-1] + '_' + TID + '.gif', interval=0.3, show_gif=False)

# What we should see is that patterns start to emerge, and with 
# higher and higher complexity as we get deeper into the network!

# Think back to when we were trying to hand engineer a convolution 
# kernel in the first session. This should seem pretty amazing to 
# you now. We've just seen that a network trained in the same way 
# that we trained on MNIST in the last session, just with more 
# layers, and a lot more data, can represent so much detail, and 
# such complex patterns.

#
# Specifying the Objective
#

# Let's summarize a bit what we've done. What we're doing is let the 
# input image's activation at some later layer or neuron determine 
# what we want to optimize. We feed an image into the network and 
# see what its activations are for a given neuron or entire layer 
# are by backproping the gradient of that activation back to the 
# input image. Remember, the gradient is just telling us how things 
# change. So by following the direction of the gradient, we're going 
# up the gradient, or ascending, and maximizing the selected layer 
# or neuron's activation by changing our input image.
#
# By going up the gradient, we're saying, let's maximize this neuron 
# or layer's activation. That's different to what we we're doing 
# with gradient descent of a cost function. Because that was a cost 
# function, we wanted to minimize it, and we were following the 
# negative direction of our gradient. So the only difference now is 
# we are following the positive direction of our gradient, and 
# performing gradient ascent.
# 
# We can also explore specifying a particular gradient activation 
# that we want to maximize. So rather than simply maximizing its 
# activation, we'll specify what we want the activation to look 
# like, and follow the gradient to get us there. For instance, let's 
# say we want to only have a particular neuron active, and nothing 
# else. We can do that by creating an array of 0s the shape of one 
# of our layers, and then filling in 1s for the output of that 
# neuron:

# Let's pick one of the later layers
layer = g.get_tensor_by_name('inception/mixed5b_pool_reduce_pre_relu:0')

# And find its shape
layer_shape = tf.shape(layer).eval(feed_dict={x:img_4d})

# We can find out how many neurons it has by feeding it an image and
# calculating the shape.  The number of output channels is the last dimension.
n_els = tf.shape(layer).eval(feed_dict={x:img_4d})[-1]

# Let's pick a random output channel
neuron_i = np.random.randint(n_els)

# And we'll create an activation of this layer which is entirely 0
layer_vec = np.zeros(layer_shape)

# Except for the randomly chosen neuron which will be full of 1s
layer_vec[..., neuron_i] = 1

# We'll go back to finding the maximal neuron in a layer
neuron = tf.reduce_max(layer, len(layer.get_shape())-1)

# And finding the mean over this neuron
gradient = tf.gradients(tf.reduce_mean(neuron), x)

# We then feed this into our feed_dict parameter and do the same 
# thing as before, ascending the gradient. We'll try this for a few 
# different neurons to see what they look like. Again, this will 
# take a long time depending on your computer!

n_iterations = 30
print("Objective, ascending the gradient...")
for i in range(5):
    print("i: ", i)
    neuron_i = np.random.randint(n_els)
    layer_vec = np.zeros(layer_shape)
    layer_vec[..., neuron_i] = 1
    img_copy = img_noise.copy() / 255.0
    imgs = []
    for it_i in range(n_iterations):
        print(it_i, end=', ')
        this_res = sess.run(gradient[0], feed_dict={
            x: img_copy,
            layer: layer_vec})[0]
        this_res /= (np.max(np.abs(this_res)) + 1e-8)
        img_copy += this_res * step
        if it_i % gif_step == 0:
            imgs.append(normalize(img_copy[0]))
    print(" build_gif")
    gif.build_gif(imgs, saveto='2-objective-' + str(neuron_i) + 
        '_' + TID + '.gif', interval=0.3, show_gif=False)

# So there is definitely something very interesting happening in 
# each of these neurons. Even though each image starts off exactly 
# the same, from the same noise image, they each end up in a very 
# different place. What we're seeing is how each neuron we've chosen 
# seems to be encoding something complex. They even somehow trigger 
# our perception in a way that says, "Oh, that sort of looks like... 
# maybe a house, or a dog, or a fish, or person...something"
#
# Since our network is trained on objects, we know what each neuron 
# of the last layer should represent. So we can actually try this 
# with the very last layer, the final layer which we know should 
# represent 1 of 1000 possible objects. Let's see how to do this. 
# Let's first find a good neuron:

print(net['labels'])

# let's try a school bus.
neuron_i = 962
print(net['labels'][neuron_i])

# We'll pick the very last layer
layer = g.get_tensor_by_name(names[-1] + ':0')

# Then find the max activation of this layer
gradient = tf.gradients(tf.reduce_max(layer), x)

# We'll find its shape and create the activation we want to maximize w/ gradient ascent
layer_shape = tf.shape(layer).eval(feed_dict={x: img_noise})
layer_vec = np.zeros(layer_shape)
layer_vec[..., neuron_i] = 1

# And then train just like before:

n_iterations = 100
gif_step = 10

img_copy = img_noise.copy()
imgs = []
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
gif.build_gif(imgs, saveto='2-object-' + str(neuron_i) + '_' + TID + '.gif', interval=0.3, show_gif=False)

# So what we should see is the noise image become more like patterns 
# that might appear on a school bus.

#
# Decaying the Gradient
#

# There is a lot we can explore with this process to get a clearer 
# picture. Some of the more interesting visualizations come about 
# through regularization techniques such as smoothing the 
# activations every so often, or clipping the gradients to a certain 
# range. We'll see how all of these together can help us get a much 
# cleaner image. We'll start with decay. This will slowly reduce the 
# range of values:

decay = 0.95

img_copy = img_noise.copy()
imgs = []
print("Decaying...")
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
print(" build_gif...")
gif.build_gif(imgs, saveto='3-decay-' + str(neuron_i) + '_' + TID + 
    '.gif', interval=0.3, show_gif=False)


#
# Blurring the Gradient
#

# Let's now try and see how blurring with a gaussian changes the 
# visualization.

# Let's get a gaussian filter
from scipy.ndimage.filters import gaussian_filter

# Which we'll smooth with a standard deviation of 0.5
sigma = 1.0

# And we'll smooth it every 4 iterations
blur_step = 5

# Now during our training, we'll smooth every blur_step iterations 
# with the given sigma.

img_copy = img_noise.copy()
imgs = []
print("Gaussian blurring...")
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay
    if it_i % blur_step == 0:
        for ch_i in range(3):
            img_copy[..., ch_i] = gaussian_filter(img_copy[..., ch_i], sigma)
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
print(" build_gif...")
gif.build_gif(imgs, saveto='4-gaussian-' + str(neuron_i) + '_' + 
    TID + '.gif', interval=0.3, show_gif=False)

# Now we're really starting to get closer to something that 
# resembles a school bus, or maybe a school bus double rainbow.


#
# Clipping the Gradient
#

# Let's now see what happens if we clip the gradient's activations:

pth = 5
img_copy = img_noise.copy()
imgs = []
print("Clipping...")
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay
    if it_i % blur_step == 0:
        for ch_i in range(3):
            img_copy[..., ch_i] = gaussian_filter(img_copy[..., ch_i], sigma)

    mask = (abs(img_copy) < np.percentile(abs(img_copy), pth))
    img_copy = img_copy - img_copy*mask

    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))

print(" build_gif...")
gif.build_gif(imgs, saveto='5-clip-' + str(neuron_i) + '_' + 
    TID + '.gif', interval=0.3, show_gif=False)

plt.title("Clipping")
plt.imshow(normalize(img_copy[0]))
plt.pause(3)


#
# Infinite Zoom / Fractal
#

# Some of the first visualizations to come out would infinitely zoom 
# into the image, creating a fractal image of the deep dream. We can 
# do this by cropping the image with a 1 pixel border, and then 
# resizing that image back to the original size.

from skimage.transform import resize
img_copy = img_noise.copy()
crop = 1
n_iterations = 1000
imgs = []
n_img, height, width, ch = img_copy.shape
print("Zoom/fractal...")
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay

    if it_i % blur_step == 0:
        for ch_i in range(3):
            img_copy[..., ch_i] = gaussian_filter(img_copy[..., ch_i], sigma)

    mask = (abs(img_copy) < np.percentile(abs(img_copy), pth))
    img_copy = img_copy - img_copy * mask

    # Crop a 1 pixel border from height and width
    img_copy = img_copy[:, crop:-crop, crop:-crop, :]

    # Resize (Note: in the lecture, we used scipy's resize which
    # could not resize images outside of 0-1 range, and so we had
    # to store the image ranges.  This is a much simpler resize
    # method that allows us to `preserve_range`.)
    img_copy = resize(img_copy[0], (height, width), order=3,
                 clip=False, preserve_range=True
                 )[np.newaxis].astype(np.float32)

    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
print(" build_gif")
gif.build_gif(imgs, saveto='6-fractal_' +  TID + '.gif',
    interval=0.3, show_gif=False)



input("End")

# eop

