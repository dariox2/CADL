

#
# Session 5, part 2
#
# CHECKPOINT WORKING!
#
# (ignore final error message when killing threads)
#


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
plt.figure(figsize=(4, 4))
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from matplotlib.cbook import MatplotlibDeprecationWarning 
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning) 


##
## Part 2 - Variational Auto-Encoding Generative Adversarial Network
##          (VAEGAN)
##

#
# In our definition of the generator, we started with a feature
# vector, `Z`. This feature vector was not connected to anything
# before it. Instead, we had to randomly create its values using a
# random number generator of its `n_latent` values from -1 to 1, and
# this range was chosen arbitrarily. It could have been 0 to 1, or -3
# to 3, or 0 to 100. In any case, the network would have had to learn
# to transform those values into something that looked like an image.
# There was no way for us to take an image, and find the feature
# vector that created it. In other words, it was not possible for us
# to *encode* an image.
#
# The closest thing to an encoding we had was taking an image and
# feeding it to the discriminator, which would output a 0 or 1. But
# what if we had another network that allowed us to encode an image,
# and then we used this network for both the discriminator and
# generative parts of the network? That's the basic idea behind the
# VAEGAN: https://arxiv.org/abs/1512.09300. It is just like the
# regular GAN, except we also use an encoder to create our feature
# vector `Z`.
#
# We then get the best of both worlds: a GAN that looks more or less
# the same, but uses the encoding from an encoder instead of an
# arbitrary feature vector; and an autoencoder that can model an
# input distribution using a trained distance function, the
# discriminator, leading to nicer encodings/decodings.
#
# Let's try to build it! Refer to the paper for the intricacies and a
# great read. Luckily, by building the `encoder` and `decoder`
# functions, we're almost there. We just need a few more components
# and will change these slightly.
#
# Let's reset our graph and recompose our network as a VAEGAN:

# In[ ]:

tf.reset_default_graph()


#
# Batch Normalization
#

# You may have noticed from the `VAE` code that I've used something
# called "batch normalization". This is a pretty effective technique
# for regularizing the training of networks by "reducing internal
# covariate shift". The basic idea is that given a minibatch, we
# optimize the gradient for this small sample of the greater
# population. But this small sample may have different
# characteristics than the entire population's gradient. Consider the
# most extreme case, a minibatch of 1. In this case, we overfit our
# gradient to optimize the gradient of the single observation. If our
# minibatch is too large, say the size of the entire population, we
# aren't able to manuvuer the loss manifold at all and the entire
# loss is averaged in a way that doesn't let us optimize anything.
# What we want to do is find a happy medium between a too-smooth loss
# surface (i.e. every observation), and a very peaky loss surface
# (i.e. a single observation). Up until now we only used mini-batches
# to help with this. But we can also approach it by "smoothing" our
# updates between each mini-batch. That would effectively smooth the
# manifold of the loss space. Those of you familiar with signal
# processing will see this as a sort of low-pass filter on the
# gradient updates.
#
# In order for us to use batch normalization, we need another
# placeholder which is a simple boolean: True or False, denoting when
# we are training. We'll use this placeholder to conditionally update
# batch normalization's statistics required for normalizing our
# minibatches. Let's create the placeholder and then I'll get into
# how to use this.

# placeholder for batch normalization
is_training = tf.placeholder(tf.bool, name='istraining')


# The original paper that introduced the idea suggests to use batch
# normalization "pre-activation", meaning after the weight
# multipllication or convolution, and before the nonlinearity. We can
# use the `libs/batch_norm` module to apply batch normalization to
# any input tensor give the tensor and the placeholder defining
# whether or not we are training. Let's use this module and you can
# inspect the code inside the module in your own time if it interests
# you.

from libs.batch_norm import batch_norm
#help(batch_norm)


# Note that Tensorflow also includes numerous batch normalization
# implementations now that it did not include at the time of filming
# (Tensorflow is evolving very quickly)! These exist in
# `tf.contrib.layers.batch_norm`, `tf.contrib.learn.ops.batch_norm`,
# and `tf.contrib.slim.batch_norm`. They work slightly differently to
# the `libs/batch_norm.py` implementation in that they take a boolean
# for whether or not you are training, rather than a
# `tf.Placeholder`. This requires you to reconstruct the network when
# you are training/inferring, or create two networks, which is
# preferable for "deploying" a model. For instance, if you have
# trained a model and you want to hand it out, you don't necessarily
# want the batch norm operations for training the network in there.
# For the libraries in this course, we'll be using the
# `libs/batch_norm` implementation which means you will have to use
# `feed_dict` to denote when you are training or not.
#


#
# Building the Encoder
#

# We can now change our encoder to accept the `is_training`
# placeholder and apply `batch_norm` just before the activation
# function is applied:

def encoder(x, is_training, channels, filter_sizes, activation=tf.nn.tanh, reuse=None):
    # Set the input to a common variable name, h, for hidden layer
    h = x

    print('encoder/input:', h.get_shape().as_list())
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
            h, W = utils.conv2d(h, channels[layer_i],
                                k_h=filter_sizes[layer_i],
                                k_w=filter_sizes[layer_i],
                                d_h=2,
                                d_w=2,
                                reuse=reuse)
            
            h = batch_norm(h, is_training)

            # Now apply the activation function
            h = activation(h)
            print('layer:', layer_i, ', shape:', h.get_shape().as_list())
            
            # Store each hidden layer
            hs.append(h)

    # Finally, return the encoding.
    return h, hs


# Let's now create the input to the network using a placeholder. We
# can try a slightly larger image this time. But be careful
# experimenting with much larger images as this is a big network.
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

n_pixels = 64
n_channels = 3
input_shape = [None, n_pixels, n_pixels, n_channels]

# placeholder for the input to the network
X = tf.placeholder(name='X', shape=input_shape, dtype=tf.float32) # dja


# And now we'll connect the input to an encoder network. We'll also
# use the `tf.nn.elu` activation instead. Explore other activations
# but I've found this to make the training much faster (e.g. 10x
# faster at least!). See the paper for more details: [Fast and
# Accurate Deep Network Learning by Exponential Linear Units (ELUs)
# ](http://arxiv.org/abs/1511.07289)
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

channels = [64, 64, 64]
filter_sizes = [5, 5, 5]
activation = tf.nn.elu
n_hidden = 128


with tf.variable_scope('encoder'):
    H, Hs = encoder(X, is_training, channels, filter_sizes, activation) # dja
    Z = utils.linear(H, n_hidden)[0]


#
# Building the Variational Layer
#

# In Session 3, we introduced the idea of Variational Bayes when we
# used the Variational Auto Encoder. The variational bayesian
# approach requires a richer understanding of probabilistic graphical
# models and bayesian methods which we we're not able to go over in
# this course (it requires a few courses all by itself!). For that
# reason, please treat this as a "black box" in this course.
#
# For those of you that are more familiar with graphical models,
# Variational Bayesian methods attempt to model an approximate joint
# distribution of $Q(Z)$ using some distance function to the true
# distribution $P(X)$. Kingma and Welling show how this approach can
# be used in a graphical model resembling an autoencoder and can be
# trained using KL-Divergence, or $KL(Q(Z) || P(X))$. The
# distribution Q(Z) is the variational distribution, and attempts to
# model the lower-bound of the true distribution $P(X)$ through the
# minimization of the KL-divergence. Another way to look at this is
# the encoder of the network is trying to model the parameters of a
# known distribution, the Gaussian Distribution, through a
# minimization of this lower bound. We assume that this distribution
# resembles the true distribution, but it is merely a simplification
# of the true distribution. To learn more about this, I highly
# recommend picking up the book by Christopher Bishop called "Pattern
# Recognition and Machine Learning" and reading the original Kingma
# and Welling paper on Variational Bayes.
#
# Now back to coding, we'll create a general variational layer that
# does exactly the same thing as our VAE in session 3. Treat this as
# a black box if you are unfamiliar with the math. It takes an input
# encoding, `h`, and an integer, `n_code` defining how many latent
# Gaussians to use to model the latent distribution. In return, we
# get the latent encoding from sampling the Gaussian layer, `z`, the
# mean and log standard deviation, as well as the prior loss,
# `loss_z`.

def variational_bayes(h, n_code):
    # Model mu and log(\sigma)
    z_mu = tf.nn.tanh(utils.linear(h, n_code, name='mu')[0])
    z_log_sigma = 0.5 * tf.nn.tanh(utils.linear(h, n_code, name='log_sigma')[0])

    # Sample from noise distribution p(eps) ~ N(0, 1)
    epsilon = tf.random_normal(tf.pack([tf.shape(h)[0], n_code]))

    # Sample from posterior
    z = z_mu + tf.mul(epsilon, tf.exp(z_log_sigma))

    # Measure loss
    loss_z = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)

    return z, z_mu, z_log_sigma, loss_z


# Let's connect this layer to our encoding, and keep all the
# variables it returns. Treat this as a black box if you are
# unfamiliar with variational bayes!
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

# Experiment w/ values between 2 - 100
# depending on how difficult the dataset is
n_code = 32

with tf.variable_scope('encoder/variational'):
    Z, Z_mu, Z_log_sigma, loss_Z = variational_bayes(h=Z, n_code=n_code)


# <a name="building-the-decoder-1"></a>
# ## Building the Decoder
#
# In the GAN network, we built a decoder and called it the generator
# network. Same idea here. We can use these terms interchangeably.
# Before we connect our latent encoding, `Z` to the decoder, we'll
# implement batch norm in our decoder just like we did with the
# encoder. This is a simple fix: add a second argument for
# `is_training` and then apply batch normalization just after the
# `deconv2d` operation and just before the nonlinear `activation`.

# In[ ]:

def decoder(z, is_training, dimensions, channels, filter_sizes,
            activation=tf.nn.elu, reuse=None):
    h = z
    for layer_i in range(len(dimensions)):
        with tf.variable_scope('layer{}'.format(layer_i+1), reuse=reuse):
            h, W = utils.deconv2d(x=h,
                               n_output_h=dimensions[layer_i],
                               n_output_w=dimensions[layer_i],
                               n_output_ch=channels[layer_i],
                               k_h=filter_sizes[layer_i],
                               k_w=filter_sizes[layer_i],
                               reuse=reuse)
            h = batch_norm(h, is_training)
            h = activation(h)
    return h


# Now we'll build a decoder just like in Session 3, and just like our
# Generator network in Part 1. In Part 1, we created `Z` as a
# placeholder which we would have had to feed in as random values.
# However, now we have an explicit coding of an input image in `X`
# stored in `Z` by having created the encoder network.

# In[ ]:

dimensions = [n_pixels // 8, n_pixels // 4, n_pixels // 2, n_pixels]
channels = [30, 30, 30, n_channels]
filter_sizes = [4, 4, 4, 4]
activation = tf.nn.elu
n_latent = n_code * (n_pixels // 16)**2

with tf.variable_scope('generator'):
    Z_decode = utils.linear(
        Z, n_output=n_latent, name='fc', activation=activation)[0]
    Z_decode_tensor = tf.reshape(
        Z_decode, [-1, n_pixels//16, n_pixels//16, n_code], name='reshape')
    G = decoder(
        Z_decode_tensor, is_training, dimensions,
        channels, filter_sizes, activation)


# Now we need to build our discriminators. We'll need to add a
# parameter for the `is_training` placeholder. We're also going to
# keep track of every hidden layer in the discriminator. Our encoder
# already returns the `Hs` of each layer. Alternatively, we could
# poll the graph for each layer in the discriminator and ask for the
# correspond layer names. We're going to need these layers when
# building our costs.

# In[ ]:

def discriminator(X,
                  is_training,
                  channels=[50, 50, 50, 50],
                  filter_sizes=[4, 4, 4, 4],
                  activation=tf.nn.elu,
                  reuse=None):

    # We'll scope these variables to "discriminator_real"
    with tf.variable_scope('discriminator', reuse=reuse):
        H, Hs = encoder(
            X, is_training, channels, filter_sizes, activation, reuse)
        shape = H.get_shape().as_list()
        H = tf.reshape(
            H, [-1, shape[1] * shape[2] * shape[3]])
        D, W = utils.linear(
            x=H, n_output=1, activation=tf.nn.sigmoid, name='fc', reuse=reuse)
    return D, Hs


# Recall the regular GAN and DCGAN required 2 discriminators: one for
# the generated samples in `Z`, and one for the input samples in `X`.
# We'll do the same thing here. One discriminator for the real input
# data, `X`, which the discriminator will try to predict as 1s, and
# another discriminator for the generated samples that go from `X`
# through the encoder to `Z`, and finally through the decoder to `G`.
# The discriminator will be trained to try and predict these as 0s,
# whereas the generator will be trained to try and predict these as
# 1s.

# In[ ]:

D_real, Hs_real = discriminator(X, is_training)
D_fake, Hs_fake = discriminator(G, is_training, reuse=True)


# <a name="building-vaegan-loss-functions"></a>
# ## Building VAE/GAN Loss Functions
#
# Let's now see how we can compose our loss. We have 3 losses for our
# discriminator. Along with measuring the binary cross entropy
# between each of them, we're going to also measure each layer's loss
# from our two discriminators using an l2-loss, and this will form
# our loss for the log likelihood measure. The details of how these
# are constructed are explained in more details in the paper:
# https://arxiv.org/abs/1512.09300 - please refer to this paper for
# more details that are way beyond the scope of this course! One
# parameter within this to pay attention to is `gamma`, which the
# authors of the paper suggest control the weighting between content
# and style, just like in Session 4's Style Net implementation.

# In[ ]:

with tf.variable_scope('loss'):
    # Loss functions
    loss_D_llike = 0
    for h_real, h_fake in zip(Hs_real, Hs_fake):
        loss_D_llike += tf.reduce_sum(tf.squared_difference(
            utils.flatten(h_fake), utils.flatten(h_real)), 1)

    eps = 1e-12
    loss_real = tf.log(D_real + eps)
    loss_fake = tf.log(1 - D_fake + eps)
    loss_GAN = tf.reduce_sum(loss_real + loss_fake, 1)
    
    gamma = 0.75
    loss_enc = tf.reduce_mean(loss_Z + loss_D_llike)
    loss_dec = tf.reduce_mean(gamma * loss_D_llike - loss_GAN)
    loss_dis = -tf.reduce_mean(loss_GAN)


# TENSORBOARD REQUIRED
#nb_utils.show_graph(tf.get_default_graph().as_graph_def())


#
# Creating the Optimizers
#

# We now have losses for our encoder, decoder, and discriminator
# networks. We can connect each of these to their own optimizer and
# start training! Just like with Part 1's GAN, we'll ensure each
# network's optimizer only trains its part of the network: the
# encoder's optimizer will only update the encoder variables, the
# generator's optimizer will only update the generator variables, and
# the discriminator's optimizer will only update the discriminator
# variables.
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

learning_rate = 0.0001

opt_enc = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(
    loss_enc,
    var_list=[var_i for var_i in tf.trainable_variables()
              if ...])

opt_gen = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(
    loss_dec,
    var_list=[var_i for var_i in tf.trainable_variables()
              if ...])

opt_dis = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(
    loss_dis,
    var_list=[var_i for var_i in tf.trainable_variables()
              if var_i.name.startswith('discriminator')])


#
# Loading the Dataset
#

# We'll now load our dataset just like in Part 1. Here is where you
# should explore with your own data!
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

from libs import datasets, dataset_utils

batch_size = 10 # 64
n_epochs = 10 # 100 dja debug
crop_shape = [n_pixels, n_pixels, n_channels]
crop_factor = 0.8
input_shape = [218, 178, 3]

# Try w/ CELEB first to make sure it works, then explore w/ your own
# dataset.
files = datasets.CELEB()
batch = dataset_utils.create_input_pipeline(
    files=files,
    batch_size=batch_size,
    n_epochs=n_epochs,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape)


# We'll also create a latent manifold just like we've done in Session
# 3 and Part 1. This is a random sampling of 4 points in the latent
# space of `Z`. We then interpolate between then to create a
# "hyper-plane" and show the decoding of 10 x 10 points on that
# hyperplane.

n_samples = 10
zs = np.random.uniform(
    -1.0, 1.0, [4, n_code]).astype(np.float32)
zs = utils.make_latent_manifold(zs, n_samples)


# Now create a session and create a coordinator to manage our queues
# for fetching data from the input pipeline and start our queue
# runners:

# We create a session to use the graph
sess = tf.Session()
init_op = tf.initialize_all_variables()

saver = tf.train.Saver()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(init_op)


# Load an existing checkpoint if it exists to continue training.

ckpt_name="vaegan.ckpt"
#if os.path.exists(ckpt_name+"-10"): # ok
#    saver.restore(sess, ckpt_name+"-10") # ok
if os.path.exists(ckpt_name): # ok
    saver.restore(sess, ckpt_name) # ok
    print("    VAEGAN model restored.")
else:
    print("    VAEGAN model not found")


# We'll also try resythesizing a test set of images. This will help
# us understand how well the encoder/decoder network is doing:
#
# dja note: watch if training, this set is changed on each run
#
n_files = len(files)
test_xs = sess.run(batch) / 255.0
if not os.path.exists('imgs'):
    os.mkdir('imgs')
m = utils.montage(test_xs, 'imgs/test_xs.png')
plt.imshow(m)
plt.pause(1)


#
# Training
#

# Almost ready for training. Let's get some variables which we'll
# need. These are the same as Part 1's training process. We'll keep
# track of `t_i` which we'll use to create images of the current
# manifold and reconstruction every so many iterations. And we'll
# keep track of the current batch number within the epoch and the
# current epoch number.

t_i = 0
batch_i = 0
epoch_i = 0


# Just like in Part 1, we'll train trying to maintain an equilibrium
# between our Generator and Discriminator networks. You should
# experiment with the margin depending on how the training proceeds.

equilibrium = 0.693
margin = 0.4


# Now we'll train! Just like Part 1, we measure the `real_cost` and
# `fake_cost`. But this time, we'll always update the encoder. Based
# on the performance of the real/fake costs, then we'll update
# generator and discriminator networks. This will take a long time to
# produce something nice, but not nearly as long as the regular GAN
# network despite the additional parameters of the encoder and
# variational networks. Be sure to monitor the reconstructions to
# understand when your network has reached the capacity of its
# learning! For reference, on Celeb Net, I would use about 5 layers
# in each of the Encoder, Generator, and Discriminator networks using
# as input a 100 x 100 image, and a minimum of 200 channels per
# layer. This network would take about 1-2 days to train on an Nvidia
# TITAN X GPU.

fig, axs = plt.subplots(1, 2, figsize=(9, 6))
print("Training...")
while epoch_i < n_epochs:
    #print("epoch_i: ", epoch_i, "(",n_files // batch_size, ")")
    #if epoch_i % (n_files // batch_size) == 0:
    if batch_i % (n_files // batch_size) == 0: # dja bugfix
        batch_i = 0
        epoch_i += 1
        print('---------- EPOCH:', epoch_i)

    batch_i += 1
    batch_xs = sess.run(batch) / 255.0
    real_cost, fake_cost, _ = sess.run([
        loss_real, loss_fake, opt_enc],
        feed_dict={
            X: batch_xs,
            is_training: True})
    real_cost = -np.mean(real_cost)
    fake_cost = -np.mean(fake_cost)

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
        sess.run(opt_gen, feed_dict={
            X: batch_xs,
            is_training: True})
    if dis_update:
        sess.run(opt_dis, feed_dict={
            X: batch_xs,
            is_training: True})

    if batch_i % 10 == 0: #50 == 0:
        print('real:', real_cost, '/ fake:', fake_cost)

        # Plot example reconstructions from latent layer
        recon = sess.run(G, feed_dict={
                Z: zs,
                is_training: False})

        recon = np.clip(recon, 0, 1)
        m1 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/manifold_%08d.png' % t_i)

        # Plot example reconstructions
        recon = sess.run(G, feed_dict={
                X: test_xs,
                is_training: False})
        recon = np.clip(recon, 0, 1)
        m2 = utils.montage(recon.reshape([-1] + crop_shape),
                'imgs/reconstruction_%08d.png' % t_i)
        
        #fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(m1)
        axs[1].imshow(m2)
        plt.show()
        plt.pause(1)
        
        t_i += 1
        

    #if batch_i % 4==0: #200 == 0:
    if batch_i ==1: # dja
        # Save the variables to disk.
        save_path = saver.save(sess, "./" + ckpt_name,
                               #global_step=batch_i,
                               #global_step=epoch_i, # dja, ok
                               write_meta_graph=False)
        print("Model saved in file: %s" % save_path)


# In[ ]:

# One of the threads has issued an exception. So let's tell all the
# threads to shutdown.
print("Coord stop...")
coord.request_stop()

# Wait until all threads have finished.
print("Coord join...")
coord.join(threads) # "-> Enqueue operation was cancelled"

# Clean up the session.
print("Sess close...")
sess.close()

# eop

