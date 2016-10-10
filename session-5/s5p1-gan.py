

### Session 5: Generative Networks
### Assignment: Generative Adversarial Networks and Recurrent Neural
### Networks

#
# Table of Contents
#

# - [Overview](#overview)
# - [Learning Goals](#learning-goals)
# - [Part 1 - Generative Adversarial Networks (GAN) / Deep
# Convolutional GAN
# (DCGAN)](#part-1---generative-adversarial-networks-gan--deep-convolutional-gan-dcgan)
# - [Introduction](#introduction)
# - [Building the Encoder](#building-the-encoder)
# - [Building the Discriminator for the Training
# Samples](#building-the-discriminator-for-the-training-samples)
# - [Building the Decoder](#building-the-decoder)
# - [Building the Generator](#building-the-generator)
# - [Building the Discriminator for the Generated
# Samples](#building-the-discriminator-for-the-generated-samples)
# - [GAN Loss Functions](#gan-loss-functions)
# - [Building the Optimizers w/
# Regularization](#building-the-optimizers-w-regularization)
# - [Loading a Dataset](#loading-a-dataset)
# - [Training](#training)
# - [Equilibrium](#equilibrium)
# - [Part 2 - Variational Auto-Encoding Generative Adversarial
# Network
# (VAEGAN)](#part-2---variational-auto-encoding-generative-adversarial-network-vaegan)
# - [Batch Normalization](#batch-normalization)
# - [Building the Encoder](#building-the-encoder-1)
# - [Building the Variational Layer](#building-the-variational-layer)
# - [Building the Decoder](#building-the-decoder-1)
# - [Building VAE/GAN Loss
# Functions](#building-vaegan-loss-functions)
# - [Creating the Optimizers](#creating-the-optimizers)
# - [Loading the Dataset](#loading-the-dataset)
# - [Training](#training-1)
# - [Part 3 - Latent-Space
# Arithmetic](#part-3---latent-space-arithmetic)
# - [Loading the Pre-Trained Model](#loading-the-pre-trained-model)
# - [Exploring the Celeb Net
# Attributes](#exploring-the-celeb-net-attributes)
# - [Find the Latent Encoding for an
# Attribute](#find-the-latent-encoding-for-an-attribute)
# - [Latent Feature Arithmetic](#latent-feature-arithmetic)
# - [Extensions](#extensions)
# - [Part 4 - Character-Level Language
# Model](session-5-part-2.ipynb#part-4---character-level-language-model)
# - [Part 5 - Pretrained Char-RNN of Donald
# Trump](session-5-part-2.ipynb#part-5---pretrained-char-rnn-of-donald-trump)
# - [Getting the Trump
# Data](session-5-part-2.ipynb#getting-the-trump-data)
# - [Basic Text Analysis](session-5-part-2.ipynb#basic-text-analysis)
# - [Loading the Pre-trained Trump
# Model](session-5-part-2.ipynb#loading-the-pre-trained-trump-model)
# - [Inference: Keeping Track of the
# State](session-5-part-2.ipynb#inference-keeping-track-of-the-state)
# - [Probabilistic
# Sampling](session-5-part-2.ipynb#probabilistic-sampling)
# - [Inference:
# Temperature](session-5-part-2.ipynb#inference-temperature)
# - [Inference: Priming](session-5-part-2.ipynb#inference-priming)
# - [Assignment
# Submission](session-5-part-2.ipynb#assignment-submission)
# <!-- /MarkdownTOC -->
#
#

#
# Overview
#

# This is certainly the hardest session and will require a lot of
# time and patience to complete. Also, many elements of this session
# may require further investigation, including reading of the
# original papers and additional resources in order to fully grasp
# their understanding. The models we cover are state of the art and
# I've aimed to give you something between a practical and
# mathematical understanding of the material, though it is a tricky
# balance. I hope for those interested, that you delve deeper into
# the papers for more understanding. And for those of you seeking
# just a practical understanding, that these notebooks will suffice.
#
# This session covered two of the most advanced generative networks:
# generative adversarial networks and recurrent neural networks.
# During the homework, we'll see how these work in more details and
# try building our own. I am not asking you train anything in this
# session as both GANs and RNNs take many days to train. However, I
# have provided pre-trained networks which we'll be exploring. We'll
# also see how a Variational Autoencoder can be combined with a
# Generative Adversarial Network to allow you to also encode input
# data, and I've provided a pre-trained model of this type of model
# trained on the Celeb Faces dataset. We'll see what this means in
# more details below.
#
# After this session, you are also required to submit your final
# project which can combine any of the materials you have learned so
# far to produce a short 1 minute clip demonstrating any aspect of
# the course you want to invesitgate further or combine with anything
# else you feel like doing. This is completely open to you and to
# encourage your peers to share something that demonstrates creative
# thinking. Be sure to keep the final project in mind while browsing
# through this notebook!
#


#
# Learning Goals
#

# * Learn to build the components of a Generative Adversarial Network
# and how it is trained
# * Learn to combine the Variational Autoencoder with a Generative
# Adversarial Network
# * Learn to use latent space arithmetic with a pre-trained VAE/GAN
# network
# * Learn to build the components of a Character Recurrent Neural
# Network and how it is trained
# * Learn to sample from a pre-trained CharRNN model


print("Begin import...")


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
#from skimage import data # ERROR: Cannot load libmkl_def.so
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
print("Loading tensorflow...")
import tensorflow as tf
from libs import utils, gif, datasets, dataset_utils, nb_utils


# dja
plt.style.use('bmh')
#import datetime
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
#plt.figure(figsize=(4, 4))
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from matplotlib.cbook import MatplotlibDeprecationWarning 
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning) 

##
## Part 1 - Generative Adversarial Networks (GAN) / 
##          Deep Convolutional GAN (DCGAN)
##


#
# Introduction
#

# Recall from the lecture that a Generative Adversarial Network is
# two networks, a generator and a discriminator. The "generator"
# takes a feature vector and decodes this feature vector to become an
# image, exactly like the decoder we built in Session 3's
# Autoencoder. The discriminator is exactly like the encoder of the
# Autoencoder, except it can only have 1 value in the final layer. We
# use a sigmoid to squash this value between 0 and 1, and then
# interpret the meaning of it as: 1, the image you gave me was real,
# or 0, the image you gave me was generated by the generator, it's a
# FAKE! So the discriminator is like an encoder which takes an image
# and then perfoms lie detection. Are you feeding me lies? Or is the
# image real?
#
# Consider the AE and VAE we trained in Session 3. The loss function
# operated partly on the input space. It said, per pixel, what is the
# difference between my reconstruction and the input image? The
# l2-loss per pixel. Recall at that time we suggested that this
# wasn't the best idea because per-pixel differences aren't
# representative of our own perception of the image. One way to
# consider this is if we had the same image, and translated it by a
# few pixels. We would not be able to tell the difference, but the
# per-pixel difference between the two images could be enormously
# high.
#
# The GAN does not use per-pixel difference. Instead, it trains a
# distance function: the discriminator. The discriminator takes in
# two images, the real image and the generated one, and learns what a
# similar image should look like! That is really the amazing part of
# this network and has opened up some very exciting potential future
# directions for unsupervised learning. Another network that also
# learns a distance function is known as the siamese network. We
# didn't get into this network in this course, but it is commonly
# used in facial verification, or asserting whether two faces are the
# same or not.
#
# The GAN network is notoriously a huge pain to train! For that
# reason, we won't actually be training it. Instead, we'll discuss an
# extension to this basic network called the VAEGAN which uses the
# VAE we created in Session 3 along with the GAN. We'll then train
# that network in Part 2. For now, let's stick with creating the GAN.
#
# Let's first create the two networks: the discriminator and the
# generator. We'll first begin by building a general purpose encoder
# which we'll use for our discriminator. Recall that we've already
# done this in Session 3. What we want is for the input placeholder
# to be encoded using a list of dimensions for each of our encoder's
# layers. In the case of a convolutional network, our list of
# dimensions should correspond to the number of output filters. We
# also need to specify the kernel heights and widths for each layer's
# convolutional network.
#
# We'll first need a placeholder. This will be the "real" image input
# to the discriminator and the discrimintator will encode this image
# into a single value, 0 or 1, saying, yes this is real, or no, this
# is not real.
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

# We'll keep a variable for the size of our image.
n_pixels = 32
n_channels = 3
input_shape = [None, n_pixels, n_pixels, n_channels]

# And then create the input image placeholder
X = tf.placeholder(name='X', shape=input_shape, dtype=tf.float32) # dja


#
# Building the Encoder
#

# Let's build our encoder just like in Session 3. We'll create a
# function which accepts the input placeholder, a list of dimensions
# describing the number of convolutional filters in each layer, and a
# list of filter sizes to use for the kernel sizes in each
# convolutional layer. We'll also pass in a parameter for which
# activation function to apply.
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

print("def encoder...")
def encoder(x, channels, filter_sizes, activation=tf.nn.tanh, reuse=None):
    # Set the input to a common variable name, h, for hidden layer
    h = x

    # Now we'll loop over the list of dimensions defining the number
    # of output filters in each layer, and collect each hidden layer
    hs = []
    for layer_i in range(len(channels)):

        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            # Convolve using the utility convolution function
            # This requirs the number of output filter,
            # and the size of the kernel in `k_h` and `k_w`.
            # By default, this will use a stride of 2, meaning
            # each new layer will be downsampled by 2.
            h, W = utils.conv2d(h,
                                #len(filter_sizes) # no
                                channels[layer_i], # fixed
                                # (both 4 here)
                                k_h=filter_sizes[layer_i],
                                k_w=filter_sizes[layer_i],
                                d_h=2, d_w=2, 
                                reuse=reuse) # dja

            # Now apply the activation function
            h = activation(h)
                                            
            # Store each hidden layer
            hs.append(h)

    # Finally, return the encoding.
    return h, hs


# 
# Building the Discriminator for the Training Samples
#

# Finally, let's take the output of our encoder, and make sure it has
# just 1 value by using a fully connected layer. We can use the
# `libs/utils` module's, `linear` layer to do this, which will also
# reshape our 4-dimensional tensor to a 2-dimensional one prior to
# using the fully connected layer.
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:
print("def discriminator...")
def discriminator(X,
                  channels=[50, 50, 50, 50],
                  filter_sizes=[4, 4, 4, 4],
                  activation=utils.lrelu,
                  reuse=None):

    # We'll scope these variables to "discriminator_real"
    with tf.variable_scope('discriminator', reuse=reuse):
        # Encode X:
        H, Hs = encoder(X, channels, filter_sizes, activation, reuse)
        
        # Now make one last layer with just 1 output.  We'll
        # have to reshape to 2-d so that we can create a fully
        # connected layer:
        shape = H.get_shape().as_list()
        H = tf.reshape(H, [-1, shape[1] * shape[2] * shape[3]])
        
        # Now we can connect our 2D layer to a single neuron output
        # w/a sigmoid activation:
        D, W = utils.linear(x=H, n_output=1,
          activation=tf.nn.sigmoid, reuse=reuse) # dja
    return D


# Now let's create the discriminator for the real training data
# coming from `X`:

# In[ ]:

D_real = discriminator(X)


# And we can see what the network looks like now:

# In[ ]:

graph = tf.get_default_graph()
nb_utils.show_graph(graph.as_graph_def())


#
# Building the Decoder
#

# Now we're ready to build the Generator, or decoding network. This
# network takes as input a vector of features and will try to produce
# an image that looks like our training data. We'll send this
# synthesized image to our discriminator which we've just built
# above.
#
# Let's start by building the input to this network. We'll need a
# placeholder for the input features to this network. We have to be
# mindful of how many features we have. The feature vector for the
# Generator will eventually need to form an image. What we can do is
# create a 1-dimensional vector of values for each element in our
# batch, giving us `[None, n_features]`. We can then reshape this to
# a 4-dimensional Tensor so that we can build a decoder network just
# like in Session 3.
#
# But how do we assign the values from our 1-d feature vector (or 2-d
# tensor with Batch number of them) to the 3-d shape of an image (or
# 4-d tensor with Batch number of them)? We have to go from the
# number of features in our 1-d feature vector, let's say `n_latent`
# to `height x width x channels` through a series of convolutional
# transpose layers. One way to approach this is think of the reverse
# process. Starting from the final decoding of `height x width x
# channels`, I will use convolution with a stride of 2, so downsample
# by 2 with each new layer. So the second to last decoder layer would
# be, `height // 2 x width // 2 x ?`. If I look at it like this, I
# can use the variable `n_pixels` denoting the `height` and `width`
# to build my decoder, and set the channels to whatever I want.
#
# Let's start with just our 2-d placeholder which will have `None x
# n_features`, then convert it to a 4-d tensor ready for the decoder
# part of the network (a.k.a. the generator).

# In[ ]:

# We'll need some variables first. This will be how many
# channels our generator's feature vector has. Experiment w/
# this if you are training your own network.
n_code = 16

# And in total how many feature it has, including the spatial
# dimensions.
n_latent = (n_pixels // 16) * (n_pixels // 16) * n_code
  # dja note: 16 = n_pixels/2 ??

# Let's build the 2-D placeholder, which is the 1-d feature vector
# for every
# element in our batch. We'll then reshape this to 4-D for the
# decoder.
Z = tf.placeholder(name='Z', shape=[None, n_latent], dtype=tf.float32)

# Now we can reshape it to input to the decoder. Here we have to
# be mindful of the height and width as described before. We need
# to make the height and width a factor of the final height and width
# that we want. Since we are using strided convolutions of 2, then
# we can say with 4 layers, that first decoder's layer should be:
# n_pixels / 2 / 2 / 2 / 2, or n_pixels / 16:
Z_tensor = tf.reshape(Z, [-1, n_pixels // 16, n_pixels // 16, n_code])


# Now we'll build the decoder in much the same way as we built our
# encoder. And exactly as we've done in Session 3! This requires one
# additional parameter "channels" which is how many output filters we
# want for each net layer. We'll interpret the `dimensions` as the
# height and width of the tensor in each new layer, the `channels` is
# how many output filters we want for each net layer, and the
# `filter_sizes` is the size of the filters used for convolution.
# We'll default to using a stride of two which will downsample each
# layer. We're also going to collect each hidden layer `h` in a list.
# We'll end up needing this for Part 2 when we combine the
# variational autoencoder w/ the generative adversarial network.

print("def decoder...")
def decoder(z, dimensions, channels, filter_sizes,
            activation=tf.nn.relu, reuse=None):
    h = z
    hs = []
    for layer_i in range(len(dimensions)):
        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            h, W = utils.deconv2d(x=h,
                               n_output_h=dimensions[layer_i],
                               n_output_w=dimensions[layer_i],
                               n_output_ch=channels[layer_i],
                               k_h=filter_sizes[layer_i],
                               k_w=filter_sizes[layer_i],
                               reuse=reuse)
            h = activation(h)
            hs.append(h)
    return h, hs


#
# Building the Generator
#

# Now we're ready to use our decoder to take in a vector of features
# and generate something that looks like our training images. We have
# to ensure that the last layer produces the same output shape as the
# discriminator's input. E.g. we used a `[None, 64, 64, 3]` input to
# the discriminator, so our generator needs to also output `[None,
# 64, 64, 3]` tensors. In other words, we have to ensure the last
# element in our `dimensions` list is 64, and the last element in our
# `channels` list is 3.

# Explore these parameters.

print("def generator...")
def generator(Z,
              dimensions=[n_pixels//8, n_pixels//4, n_pixels//2, n_pixels],
              channels=[50, 50, 50, n_channels],
              filter_sizes=[4, 4, 4, 4],
              activation=utils.lrelu):

    with tf.variable_scope('generator'):
        G, Hs = decoder(Z_tensor, dimensions, channels, filter_sizes, activation)

    return G


# Now let's call the `generator` function with our input placeholder
# `Z`. This will take our feature vector and generate something in
# the shape of an image.

# In[ ]:

G = generator(Z)


# In[ ]:

graph = tf.get_default_graph()
nb_utils.show_graph(graph.as_graph_def())


#
# Building the Discriminator for the Generated Samples
#

# Lastly, we need *another* discriminator which takes as input our
# generated images. Recall the discriminator that we have made only
# takes as input our placeholder `X` which is for our actual training
# samples. We'll use the same function for creating our discriminator
# and **reuse** the variables we already have. This is the crucial
# part! We aren't making *new* trainable variables, but reusing the
# ones we have. We're just create a new set of operations that takes
# as input our generated image. So we'll have a whole new set of
# operations exactly like the ones we have created for our first
# discriminator. But we are going to use the exact same variables as
# our first discriminator, so that we optimize the same values.

# In[ ]:

D_fake = discriminator(G, reuse=True)


# Now we can look at the graph and see the new discriminator inside
# the node for the discriminator. You should see the original
# discriminator and a new graph of a discriminator within it, but all
# the weights are shared with the original discriminator.

# In[ ]:

nb_utils.show_graph(graph.as_graph_def())


#
# GAN Loss Functions
#

# We now have all the components to our network. We just have to
# train it. This is the notoriously tricky bit. We will have 3
# different loss measures instead of our typical network with just a
# single loss. We'll later connect each of these loss measures to two
# optimizers, one for the generator and another for the
# discriminator, and then pin them against each other and see which
# one wins! Exciting times!
#
# Recall from Session 3's Supervised Network, we created a binary
# classification task: music or speech. We again have a binary
# classification task: real or fake. So our loss metric will again
# use the binary cross entropy to measure the loss of our three
# different modules: the generator, the discriminator for our real
# images, and the discriminator for our generated images.
#
# To find out the loss function for our generator network, answer the
# question, what makes the generator successful? Successfully fooling
# the discriminator. When does that happen? When the discriminator
# for the fake samples produces all ones. So our binary cross entropy
# measure will measure the cross entropy with our predicted
# distribution and the true distribution which has all ones.

print("create loss_G...")
with tf.variable_scope('loss/generator'):
    loss_G = tf.reduce_mean(utils.binary_cross_entropy(D_fake, tf.ones_like(D_fake)))


# What we've just written is a loss function for our generator. The
# generator is optimized when the discriminator for the generated
# samples produces all ones. In contrast to the generator, the
# discriminator will have 2 measures to optimize. One which is the
# opposite of what we have just written above, as well as 1 more
# measure for the real samples. Try writing these two losses and
# we'll combine them using their average. We want to optimize the
# Discriminator for the real samples producing all 1s, and the
# Discriminator for the fake samples producing all 0s:
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


print("create loss_D...")
with tf.variable_scope('loss/discriminator/real'):
    loss_D_real = utils.binary_cross_entropy(D_real, tf.ones_like(D_fake)) # dja
with tf.variable_scope('loss/discriminator/fake'):
    loss_D_fake = utils.binary_cross_entropy(D_fake, tf.zeros_like(D_fake)) # dja
with tf.variable_scope('loss/discriminator'):
    loss_D = tf.reduce_mean((loss_D_real + loss_D_fake) / 2)


nb_utils.show_graph(graph.as_graph_def())


# With our loss functions, we can create an optimizer for the
# discriminator and generator:

#
# Building the Optimizers w/ Regularization
#

# We're almost ready to create our optimizers. We just need to do one
# extra thing. Recall that our loss for our generator has a flow from
# the generator through the discriminator. If we are training both
# the generator and the discriminator, we have two measures which
# both try to optimize the discriminator, but in opposite ways: the
# generator's loss would try to optimize the discriminator to be bad
# at its job, and the discriminator's loss would try to optimize it
# to be good at its job. This would be counter-productive, trying to
# optimize opposing losses. What we want is for the generator to get
# better, and the discriminator to get better. Not for the
# discriminator to get better, then get worse, then get better,
# etc... The way we do this is when we optimize our generator, we let
# the gradient flow through the discriminator, but we do not update
# the variables in the discriminator. Let's try and grab just the
# discriminator variables and just the generator variables below:
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# dja debug
#for v in tf.trainable_variables():
#    print("  ->",v.name)

# Grab just the variables corresponding to the discriminator
# and just the generator:
vars_d = [v for v in tf.trainable_variables()
             if v.name.startswith("discriminator/")] # dja
print('Training discriminator variables:')
[print(v.name) for v in tf.trainable_variables()
 if v.name.startswith('discriminator')]

vars_g = [v for v in tf.trainable_variables()
             if v.name.startswith('generator/')] # dja
print('Training generator variables:')
[print(v.name) for v in tf.trainable_variables()
 if v.name.startswith('generator')]


# We can also apply regularization to our network. This will penalize
# weights in the network for growing too large.

# In[ ]:

d_reg = tf.contrib.layers.apply_regularization(
    tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(
    tf.contrib.layers.l2_regularizer(1e-6), vars_g)


# The last thing you may want to try is creating a separate learning
# rate for each of your generator and discriminator optimizers like
# so:

# In[ ]:

learning_rate = 0.0001

lr_g = tf.placeholder(tf.float32, shape=[], name='learning_rate_g')
lr_d = tf.placeholder(tf.float32, shape=[], name='learning_rate_d')


# Now you can feed the placeholders to your optimizers. If you run
# into errors creating these, then you likely have a problem with
# your graph's definition! Be sure to go back and reset the default
# graph and check the sizes of your different
# operations/placeholders.
#
# With your optimizers, you can now train the network by "running"
# the optimizer variables with your session. You'll need to set the
# `var_list` parameter of the `minimize` function to only train the
# variables for the discriminator and same for the generator's
# optimizer:
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

opt_g = tf.train.AdamOptimizer(learning_rate=lr_g).minimize(loss_G + g_reg, var_list=vars_g) # dja


# In[ ]:

opt_d = tf.train.AdamOptimizer(learning_rate=lr_d).minimize(loss_D + d_reg, var_list=vars_d)


#
# Loading a Dataset
#

# Let's use the Celeb Dataset just for demonstration purposes. In
# Part 2, you can explore using your own dataset. This code is
# exactly the same as we did in Session 3's homework with the VAE.
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

# You'll want to change this to your own data if you end up training
# your own GAN.
batch_size = 16 # 64 # I have 100 samples; 64 does not work
n_epochs = 100 # 1
crop_shape = [n_pixels, n_pixels, 3]
crop_factor = 0.8
input_shape = [218, 178, 3]

files = datasets.CELEB()
print("Creating input pipeline CELEB, len=", len(files))
batch = dataset_utils.create_input_pipeline(
        files=files,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape)


#
# Training
#
# We'll now go through the setup of training the network. We won't
# actually spend the time to train the network but just see how it
# would be done. This is because in Part 2, we'll see an extension to
# this network which makes it much easier to train.

# dja note: F****N SAVER DOES NOT WORK HERE EITHER

ckpt_name = 'gan.ckpt'

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
coord = tf.train.Coordinator()
tf.get_default_graph().finalize()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

if os.path.exists(ckpt_name):
    saver.restore(sess, ckpt_name)
    print("VAE model restored.")


n_examples = 10

zs = np.random.uniform(0.0, 1.0, [4, n_latent]).astype(np.float32)
zs = utils.make_latent_manifold(zs, n_examples)


#
# Equilibrium
#

# Equilibrium is at 0.693. Why? Consider what the cost is measuring,
# the binary cross entropy. If we have random guesses, then we have
# as many 0s as we have 1s. And on average, we'll be 50% correct. The
# binary cross entropy is:
#
# \begin{align}
# \sum_i \text{X}_i * \text{log}(\tilde{\text{X}}_i) + (1 -
# \text{X}_i) * \text{log}(1 - \tilde{\text{X}}_i)
# \end{align}
#
# Which is written out in tensorflow as:
#
#    (-(x * tf.log(z) + (1. - x) * tf.log(1. - z)))
#
# Where `x` is the discriminator's prediction of the true
# distribution, in the case of GANs, the input images, and `z` is the
# discriminator's prediction of the generated images corresponding to
# the mathematical notation of $\tilde{\text{X}}$. We sum over all
# features, but in the case of the discriminator, we have just 1
# feature, the guess of whether it is a true image or not. If our
# discriminator guesses at chance, i.e. 0.5, then we'd have something
# like:
#
# \begin{align}
# 0.5 * \text{log}(0.5) + (1 - 0.5) * \text{log}(1 - 0.5) = -0.693
# \end{align}
#
# So this is what we'd expect at the start of learning and from a
# game theoretic point of view, where we want things to remain. So
# unlike our previous networks, where our loss continues to drop
# closer and closer to 0, we want our loss to waver around this value
# as much as possible, and hope for the best.

# In[ ]:

equilibrium = 0.693
margin = 0.2


# When we go to train the network, we switch back and forth between
# each optimizer, feeding in the appropriate values for each
# optimizer. The `opt_g` optimizer only requires the `Z` and `lr_g`
# placeholders, while the `opt_d` optimizer requires the `X`, `Z`,
# and `lr_d` placeholders.
#
# Don't train this network for very long because GANs are a huge pain
# to train and require a lot of fiddling. They very easily get stuck
# in their adversarial process, or get overtaken by one or the other,
# resulting in a useless model. What you need to develop is a steady
# equilibrium that optimizes both. That will likely take two weeks
# just trying to get the GAN to train and not have enough time for
# the rest of the assignment. They require a lot of memory/cpu and
# can take many days to train once you have settled on an
# architecture/training process/dataset. Just let it run for a short
# time and then interrupt the kernel (don't restart!), then continue
# to the next cell.
#
# From there, we'll go over an extension to the GAN which uses a VAE
# like we used in Session 3. By using this extra network, we can
# actually train a better model in a fraction of the time and with
# much more ease! But the network's definition is a bit more
# complicated. Let's see how the GAN is trained first and then we'll
# train the VAE/GAN network instead. While training, the "real" and
# "fake" cost will be printed out. See how this cost wavers around
# the equilibrium and how we enforce it to try and stay around there
# by including a margin and some simple logic for updates. This is
# highly experimental and the research does not have a good answer
# for the best practice on how to train a GAN. I.e., some people will
# set the learning rate to some ratio of the performance between
# fake/real networks, others will have a fixed update schedule but
# train the generator twice and the discriminator only once.

# In[ ]:

t_i = 0
batch_i = 0
epoch_i = 0
n_files = len(files)
fig, axs = plt.subplots(1, 2, figsize=(9, 6))
while epoch_i < n_epochs:

    batch_i += 1
    batch_xs = sess.run(batch) / 255.0
    batch_zs = np.random.uniform(
        0.0, 1.0, [batch_size, n_latent]).astype(np.float32)

    real_cost, fake_cost = sess.run([
        loss_D_real, loss_D_fake],
        feed_dict={
            X: batch_xs,
            Z: batch_zs})
    real_cost = np.mean(real_cost)
    fake_cost = np.mean(fake_cost)
    
    if (batch_i % 2) == 0: # batch_i % 20
        print(batch_i, 'real:', real_cost, '/ fake:', fake_cost)

    gen_update = True
    dis_update = True

    if real_cost > (equilibrium + margin) or        fake_cost > (equilibrium + margin):
        gen_update = False

    if real_cost < (equilibrium - margin) or        fake_cost < (equilibrium - margin):
        dis_update = False

    if not (gen_update or dis_update):
        gen_update = True
        dis_update = True

    if gen_update:
        sess.run(opt_g,
            feed_dict={
                Z: batch_zs,
                lr_g: learning_rate})
    if dis_update:
        sess.run(opt_d,
            feed_dict={
                X: batch_xs,
                Z: batch_zs,
                lr_d: learning_rate})

    if batch_i % (n_files // batch_size) == 0:
        batch_i = 0
        epoch_i += 1
        print('---------- EPOCH:', epoch_i)
        
        # Plot example reconstructions from latent layer
        recon = sess.run(G, feed_dict={Z: zs})

        recon = np.clip(recon, 0, 1)
        m1 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/manifold_%08d.png' % t_i)

        recon = sess.run(G, feed_dict={Z: batch_zs})

        recon = np.clip(recon, 0, 1)
        m2 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/reconstructions_%08d.png' % t_i)
        
        #fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(m1)
        axs[1].imshow(m2)
        plt.show()
        plt.pause(1)

        t_i += 1

        # Save the variables to disk.
        save_path = saver.save(sess, "./" + ckpt_name,
                               global_step=batch_i,
                               write_meta_graph=False)
        print("Model saved in file: %s" % save_path)


# In[ ]:

# Tell all the threads to shutdown.
coord.request_stop()

# Wait until all threads have finished.
coord.join(threads)

# Clean up the session.
sess.close()

# eop part 1

