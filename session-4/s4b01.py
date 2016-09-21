
print("Begin...")
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
#plt.figure(figsize=(5, 5))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")


sess = tf.InteractiveSession()

from libs import inception
print("Loading inception...")
net = inception.get_inception_model()

# REQUIERE TENSORBOARD
#nb_utils.show_graph(net['graph_def'])

tf.import_graph_def(net['graph_def'], name='inception')

print("net labels: ", net['labels'])

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
plt.imshow(og)
#plt.show()
plt.title("coffee")
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

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img)

# Note that unlike the lecture, we have to call the `inception.deprocess`
# function so that it adds back the mean!
plt.title("deprocess")
axs[1].imshow(inception.deprocess(img))
#plt.show()
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
plt.figure(figsize=(10,10))
plt.imshow(W_montage, interpolation='nearest')
plt.pause(5)

# Or, we can also try to look at them as RGB filters, showing the
# influence of each color channel, for each neuron or output filter.

Ws = [utils.montage_filters(W_eval[:, :, [i], :]) for i in range(3)]
Ws = np.rollaxis(np.array(Ws), 0, 3)
plt.figure(figsize=(10,10))
plt.imshow(Ws, interpolation='nearest')
plt.pause(5)

# In order to better see what these are doing, let's normalize the
# filters range:

np.min(Ws), np.max(Ws)
Ws = (Ws / np.max(np.abs(Ws)) * 128 + 128).astype(np.uint8)
plt.figure(figsize=(10,10))
plt.imshow(Ws, interpolation='nearest')
plt.pause(5)

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
fig, axs = plt.subplots(1, 3, figsize=(20, 10))
axs[0].imshow(inception.deprocess(img))
axs[0].set_title('Original Image')
axs[1].imshow(Ws, interpolation='nearest')
axs[1].set_title('Convolution Filters')
axs[2].imshow(montage, cmap='gray')
axs[2].set_title('Convolution Outputs')
plt.pause(5)

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

fig, axs = plt.subplots(1, 2)
axs[0].imshow(inception.deprocess(img))
axs[1].imshow(res[0])

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
fig, axs = plt.subplots(1, 2)
axs[0].imshow(inception.deprocess(img))
axs[1].imshow(r[0])

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

plt.figure(figsize=(12, 12))
plt.imshow(montage)
plt.pause(5)

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
fig, axs = plt.subplots(1, n_plots, figsize=(20, 5))
base = img_4d
axs[0].imshow(inception.deprocess(img))
for feature_i, featurename in enumerate(features):
    feature = g.get_tensor_by_name(featurename + ':0')
    neuron = tf.reduce_max(feature, len(feature.get_shape())-1)
    gradient = tf.gradients(tf.reduce_sum(neuron), x)
    this_res = sess.run(gradient[0], feed_dict={x: base})[0]
    axs[feature_i+1].imshow(normalize(this_res))
    axs[feature_i+1].set_title(featurename)

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

# Build the gif
gif.build_gif(imgs, saveto='1-simplest-mean-layer_' + TID + '.gif')


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
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))

gif.build_gif(imgs, saveto='1-simplest-max-neuron_' + TID + '.gif')

# What we should see here is how the maximal neuron in a layer's
# activation is slowly maximized through gradient ascent. So over time,
# we're increasing the overall activation of the neuron we asked for.
#
# Let's try doing this for each of our max pool layers, in increasing
# depth, and let it run a little longer. This will take a long time
# depending on your machine!

# For each max pooling feature, we'll produce a GIF
for feature_i in features:
    print("gif feature ", feature_i)
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
    gif.build_gif(
        imgs, saveto='1-simplest-' + feature_i.split('/')[-1] + '_' + TID + '.gif')

input("End")

# eop

