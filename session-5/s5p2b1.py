
# coding: utf-8

# # Session 5: Generative Networks
# ## Assignment: Generative Adversarial Networks, Variational
# Autoencoders, and Recurrent Neural Networks
# <p class="lead">
# <a
# href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative
# Applications of Deep Learning with Google's Tensorflow</a><br />
# <a href="http://pkmital.com">Parag K. Mital</a><br />
# <a href="https://www.kadenze.com">Kadenze, Inc.</a>
# </p>
#
#
# Continued from [session-5-part-1.ipynb](session-5-part-1.ipynb)...
#
# # Table of Contents
#
# <!-- MarkdownTOC autolink="true" autoanchor="true" bracket="round"
# -->
# - [Overview](session-5-part-1.ipynb#overview)
# - [Learning Goals](session-5-part-1.ipynb#learning-goals)
# - [Part 1 - Generative Adversarial Networks \(GAN\) / Deep
# Convolutional GAN
# \(DCGAN\)](#part-1---generative-adversarial-networks-gan--deep-convolutional-gan-dcgan)
# - [Introduction](session-5-part-1.ipynb#introduction)
# - [Building the
# Encoder](session-5-part-1.ipynb#building-the-encoder)
# - [Building the Discriminator for the Training
# Samples](session-5-part-1.ipynb#building-the-discriminator-for-the-training-samples)
# - [Building the
# Decoder](session-5-part-1.ipynb#building-the-decoder)
# - [Building the
# Generator](session-5-part-1.ipynb#building-the-generator)
# - [Building the Discriminator for the Generated
# Samples](session-5-part-1.ipynb#building-the-discriminator-for-the-generated-samples)
# - [GAN Loss Functions](session-5-part-1.ipynb#gan-loss-functions)
# - [Building the Optimizers w/
# Regularization](session-5-part-1.ipynb#building-the-optimizers-w-regularization)
# - [Loading a Dataset](session-5-part-1.ipynb#loading-a-dataset)
# - [Training](session-5-part-1.ipynb#training)
# - [Equilibrium](session-5-part-1.ipynb#equilibrium)
# - [Part 2 - Variational Auto-Encoding Generative Adversarial
# Network
# \(VAEGAN\)](#part-2---variational-auto-encoding-generative-adversarial-network-vaegan)
# - [Batch Normalization](session-5-part-1.ipynb#batch-normalization)
# - [Building the
# Encoder](session-5-part-1.ipynb#building-the-encoder-1)
# - [Building the Variational
# Layer](session-5-part-1.ipynb#building-the-variational-layer)
# - [Building the
# Decoder](session-5-part-1.ipynb#building-the-decoder-1)
# - [Building VAE/GAN Loss
# Functions](session-5-part-1.ipynb#building-vaegan-loss-functions)
# - [Creating the
# Optimizers](session-5-part-1.ipynb#creating-the-optimizers)
# - [Loading the Dataset](session-5-part-1.ipynb#loading-the-dataset)
# - [Training](session-5-part-1.ipynb#training-1)
# - [Part 3 - Latent-Space
# Arithmetic](session-5-part-1.ipynb#part-3---latent-space-arithmetic)
# - [Loading the Pre-Trained
# Model](session-5-part-1.ipynb#loading-the-pre-trained-model)
# - [Exploring the Celeb Net
# Attributes](session-5-part-1.ipynb#exploring-the-celeb-net-attributes)
# - [Find the Latent Encoding for an
# Attribute](session-5-part-1.ipynb#find-the-latent-encoding-for-an-attribute)
# - [Latent Feature
# Arithmetic](session-5-part-1.ipynb#latent-feature-arithmetic)
# - [Extensions](session-5-part-1.ipynb#extensions)
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
#
# <!-- /MarkdownTOC -->

# In[ ]:

# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n',
          'You should consider updating to Python 3.4.0 or',
          'higher as the libraries built for this course',
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda'
          'and then restart `jupyter notebook`:\n',
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    from scipy.ndimage.filters import gaussian_filter
    import IPython.display as ipyd
    import tensorflow as tf
    from libs import utils, gif, datasets, dataset_utils, nb_utils
except ImportError as e:
    print("Make sure you have started notebook in the same directory",
          "as the provided zip file which includes the 'libs' folder",
          "and the file 'utils.py' inside of it.  You will NOT be able",
          "to complete this assignment unless you restart jupyter",
          "notebook inside the directory created by extracting",
          "the zip file or cloning the github repo.")
    print(e)

# We'll tell matplotlib to inline any drawn figures like so:
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')


# In[ ]:

# Bit of formatting because I don't like the default inline code
# style:
from IPython.core.display import HTML
HTML("""<style> .rendered_html code { 
    padding: 2px 4px;
    color: #c7254e;
    background-color: #f9f2f4;
    border-radius: 4px;
} </style>""")


# <style> .rendered_html code {
# padding: 2px 4px;
# color: #c7254e;
# background-color: #f9f2f4;
# border-radius: 4px;
# } </style>
#
#
#
# <a name="part-4---character-level-language-model"></a>
# # Part 4 - Character-Level Language Model
#
# We'll now continue onto the second half of the homework and explore
# recurrent neural networks. We saw one potential application of a
# recurrent neural network which learns letter by letter the content
# of a text file. We were then able to synthesize from the model to
# produce new phrases. Let's try to build one. Replace the code below
# with something that loads your own text file or one from the
# internet. Be creative with this!
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

import tensorflow as tf
from six.moves import urllib
script = 'http://www.awesomefilm.com/script/biglebowski.txt'
txts = []
f, _ = urllib.request.urlretrieve(script, script.split('/')[-1])
with open(f, 'r') as fp:
    txt = fp.read()


# Let's take a look at the first part of this:

# In[ ]:

txt[:100]


# We'll just clean up the text a little. This isn't necessary, but
# can help the training along a little. In the example text I
# provided, there is a lot of white space (those \t's are tabs). I'll
# remove them. There are also repetitions of \n, new lines, which are
# not necessary. The code below will remove the tabs, ending
# whitespace, and any repeating newlines. Replace this with any
# preprocessing that makes sense for your dataset. Try to boil it
# down to just the possible letters for what you want to
# learn/synthesize while retaining any meaningful patterns:

# In[ ]:

txt = "\n".join([txt_i.strip()
                 for txt_i in txt.replace('\t', '').split('\n')
                 if len(txt_i)])


# Now we can see how much text we have:

# In[ ]:

len(txt)


# In general, we'll want as much text as possible. But I'm including
# this just as a minimal example so you can explore your own. Try
# making a text file and seeing the size of it. You'll want about 1
# MB at least.
#
# Let's now take a look at the different characters we have in our
# file:

# In[ ]:

vocab = list(set(txt))
vocab.sort()
len(vocab)
print(vocab)


# And then create a mapping which can take us from the letter to an
# integer look up table of that letter (and vice-versa). To do this,
# we'll use an `OrderedDict` from the `collections` library. In
# Python 3.6, this is the default behavior of dict, but in earlier
# versions of Python, we'll need to be explicit by using OrderedDict.

# In[ ]:

from collections import OrderedDict

encoder = OrderedDict(zip(vocab, range(len(vocab))))
decoder = OrderedDict(zip(range(len(vocab)), vocab))


# In[ ]:

encoder


# We'll store a few variables that will determine the size of our
# network. First, `batch_size` determines how many sequences at a
# time we'll train on. The `seqence_length` parameter defines the
# maximum length to unroll our recurrent network for. This is
# effectively the depth of our network during training to help guide
# gradients along. Within each layer, we'll have `n_cell` LSTM units,
# and `n_layers` layers worth of LSTM units. Finally, we'll store the
# total number of possible characters in our data, which will
# determine the size of our one hot encoding (like we had for MNIST
# in Session 3).

# In[ ]:

# Number of sequences in a mini batch
batch_size = 100

# Number of characters in a sequence
sequence_length = 50

# Number of cells in our LSTM layer
n_cells = 128

# Number of LSTM layers
n_layers = 3

# Total number of characters in the one-hot encoding
n_chars = len(vocab)


# Let's now create the input and output to our network. We'll use
# placeholders and feed these in later. The size of these need to be
# [`batch_size`, `sequence_length`]. We'll then see how to build the
# network in between.
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

X = tf.placeholder(tf.int32, shape=..., name='X')

# We'll have a placeholder for our true outputs
Y = tf.placeholder(tf.int32, shape=..., name='Y')


# The first thing we need to do is convert each of our
# `sequence_length` vectors in our batch to `n_cells` LSTM cells. We
# use a lookup table to find the value in `X` and use this as the
# input to `n_cells` LSTM cells. Our lookup table has `n_chars`
# possible elements and connects each character to `n_cells` cells.
# We create our lookup table using `tf.get_variable` and then the
# function `tf.nn.embedding_lookup` to connect our `X` placeholder to
# `n_cells` number of neurons.

# In[ ]:

# we first create a variable to take us from our one-hot
# representation to our LSTM cells
embedding = tf.get_variable("embedding", [n_chars, n_cells])

# And then use tensorflow's embedding lookup to look up the ids in X
Xs = tf.nn.embedding_lookup(embedding, X)

# The resulting lookups are concatenated into a dense tensor
print(Xs.get_shape().as_list())


# Now recall from the lecture that recurrent neural networks share
# their weights across timesteps. So we don't want to have one large
# matrix with every timestep, but instead separate them. We'll use
# `tf.split` to split our `[batch_size, sequence_length, n_cells]`
# array in `Xs` into a list of `sequence_length` elements each
# composed of `[batch_size, n_cells]` arrays. This gives us
# `sequence_length` number of arrays of `[batch_size, 1, n_cells]`.
# We then use `tf.squeeze` to remove the 1st index corresponding to
# the singleton `sequence_length` index, resulting in simply
# `[batch_size, n_cells]`.

# In[ ]:

with tf.name_scope('reslice'):
    Xs = [tf.squeeze(seq, [1])
          for seq in tf.split(1, sequence_length, Xs)]


# With each of our timesteps split up, we can now connect them to a
# set of LSTM recurrent cells. We tell the
# `tf.nn.rnn_cell.BasicLSTMCell` method how many cells we want, i.e.
# how many neurons there are, and we also specify that our state will
# be stored as a tuple. This state defines the internal state of the
# cells as well as the connection from the previous timestep. We can
# also pass a value for the `forget_bias`. Be sure to experiment with
# this parameter as it can significantly effect performance (e.g.
# Gers, Felix A, Schmidhuber, Jurgen, and Cummins, Fred. Learning to
# forget: Continual prediction with lstm. Neural computation,
# 12(10):2451–2471, 2000).

# In[ ]:

cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_cells, state_is_tuple=True, forget_bias=1.0)


# Let's take a look at the cell's state size:

# In[ ]:

cells.state_size


# `c` defines the internal memory and `h` the output. We'll have as
# part of our `cells`, both an `initial_state` and a `final_state`.
# These will become important during inference and we'll see how
# these work more then. For now, we'll set the `initial_state` to all
# zeros using the convenience function provided inside our `cells`
# object, `zero_state`:

# In[ ]:

initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)


# Looking at what this does, we can see that it creates a `tf.Tensor`
# of zeros for our `c` and `h` states for each of our `n_cells` and
# stores this as a tuple inside the `LSTMStateTuple` object:

# In[ ]:

initial_state


# So far, we have created a single layer of LSTM cells composed of
# `n_cells` number of cells. If we want another layer, we can use the
# `tf.nn.rnn_cell.MultiRNNCell` method, giving it our current cells,
# and a bit of pythonery to multiply our cells by the number of
# layers we want. We'll then update our `initial_state` variable to
# include the additional cells:

# In[ ]:

cells = tf.nn.rnn_cell.MultiRNNCell(
    [cells] * n_layers, state_is_tuple=True)
initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)


# Now if we take a look at our `initial_state`, we should see one
# `LSTMStateTuple` for each of our layers:

# In[ ]:

initial_state


# So far, we haven't connected our recurrent cells to anything. Let's
# do this now using the `tf.nn.rnn` method. We also pass it our
# `initial_state` variables. It gives us the `outputs` of the rnn, as
# well as their states after having been computed. Contrast that with
# the `initial_state`, which set the LSTM cells to zeros. After
# having computed something, the cells will all have a different
# value somehow reflecting the temporal dynamics and expectations of
# the next input. These will be stored in the `state` tensors for
# each of our LSTM layers inside a `LSTMStateTuple` just like the
# `initial_state` variable.

# In[ ]:

help(tf.nn.rnn)


# In[ ]:

Help on function rnn in module tensorflow.python.ops.rnn:

rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)
    Creates a recurrent neural network specified by RNNCell `cell`.
    
    The simplest form of RNN network generated is:
    ```py
      state = cell.zero_state(...)
      outputs = []
      for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
      return (outputs, state)
    ```
    However, a few other options are available:
    
    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.
    
    The dynamic calculation performed is, at time t for batch row b,
      (output, state)(b, t) =
        (t >= sequence_length(b))
          get_ipython().magic('pinfo ')
          : cell(input(b, t), state(b, t - 1))
    
    Args:
      cell: An instance of RNNCell.
      inputs: A length T list of inputs, each a `Tensor` of shape
        `[batch_size, input_size]`, or a nested tuple of such elements.
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
      sequence_length: Specifies the length of each sequence in inputs.
        An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
      scope: VariableScope for the created subgraph; defaults to "RNN".
    
    Returns:
      A pair (outputs, state) where:
        - outputs is a length T list of outputs (one for each input), or a nested
          tuple of such elements.
        - state is the final state
    
    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If `inputs` is `None` or an empty list, or if the input depth
        (column size) cannot be inferred from inputs via shape inference.



# Use the help on the functino `tf.nn.rnn` to create the `outputs`
# and `states` variable as below. We've already created each of the
# variable you need to use:
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>

# In[ ]:

outputs, state = tf.nn.rnn(cell=..., input=..., initial_state=...)


# Let's take a look at the state now:

# In[ ]:

state


# Our outputs are returned as a list for each of our timesteps:

# In[ ]:

outputs


# We'll now stack all our outputs for every timestep. We can treat
# every observation at each timestep and for each batch using the
# same weight matrices going forward, since these should all have
# shared weights. Each timstep for each batch is its own observation.
# So we'll stack these in a 2d matrix so that we can create our
# softmax layer:

# In[ ]:

outputs_flat = tf.reshape(tf.concat(1, outputs), [-1, n_cells])


# Our outputs are now concatenated so that we have [`batch_size *
# timesteps`, `n_cells`]

# In[ ]:

outputs_flat


# We now create a softmax layer just like we did in Session 3 and in
# Session 3's homework. We multiply our final LSTM layer's `n_cells`
# outputs by a weight matrix to give us `n_chars` outputs. We then
# scale this output using a `tf.nn.softmax` layer so that they become
# a probability by exponentially scaling its value and dividing by
# its sum. We store the softmax probabilities in `probs` as well as
# keep track of the maximum index in `Y_pred`:

# In[ ]:

with tf.variable_scope('prediction'):
    W = tf.get_variable(
        "W",
        shape=[n_cells, n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable(
        "b",
        shape=[n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))

    # Find the output prediction of every single character in our minibatch
    # we denote the pre-activation prediction, logits.
    logits = tf.matmul(outputs_flat, W) + b

    # We get the probabilistic version by calculating the softmax of this
    probs = tf.nn.softmax(logits)

    # And then we can find the index of maximum probability
    Y_pred = tf.argmax(probs, 1)


# To train the network, we'll measure the loss between our predicted
# outputs and true outputs. We could use the `probs` variable, but we
# can also make use of `tf.nn.softmax_cross_entropy_with_logits`
# which will compute the softmax for us. We therefore need to pass in
# the variable just before the softmax layer, denoted as `logits`
# (unscaled values). This takes our variable `logits`, the unscaled
# predicted outputs, as well as our true outputs, `Y`. Before we give
# it `Y`, we'll need to reshape our true outputs in the same way,
# [`batch_size` x `timesteps`, `n_chars`]. Luckily, tensorflow
# provides a convenience for doing this, the
# `tf.nn.sparse_softmax_cross_entropy_with_logits` function:

# ```python
# help(tf.nn.sparse_softmax_cross_entropy_with_logits)
#
# Help on function sparse_softmax_cross_entropy_with_logits in module
# tensorflow.python.ops.nn_ops:
#
# sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)
# Computes sparse softmax cross entropy between `logits` and
# `labels`.
#
# Measures the probability error in discrete classification tasks in
# which the
# classes are mutually exclusive (each entry is in exactly one
# class). For
# example, each CIFAR-10 image is labeled with one and only one
# label: an image
# can be a dog or a truck, but not both.
#
# **NOTE:** For this operation, the probability of a given label is
# considered
# exclusive. That is, soft classes are not allowed, and the `labels`
# vector
# must provide a single specific index for the true class for each
# row of
# `logits` (each minibatch entry). For soft softmax classification
# with
# a probability distribution for each entry, see
# `softmax_cross_entropy_with_logits`.
#
# **WARNING:** This op expects unscaled logits, since it performs a
# softmax
# on `logits` internally for efficiency. Do not call this op with the
# output of `softmax`, as it will produce incorrect results.
#
# A common use case is to have logits of shape `[batch_size,
# num_classes]` and
# labels of shape `[batch_size]`. But higher dimensions are
# supported.
#
# Args:
# logits: Unscaled log probabilities of rank `r` and shape
# `[d_0, d_1, ..., d_{r-2}, num_classes]` and dtype `float32` or
# `float64`.
# labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-2}]` and dtype
# `int32` or
# `int64`. Each entry in `labels` must be an index in `[0,
# num_classes)`.
# Other values will result in a loss of 0, but incorrect gradient
# computations.
# name: A name for the operation (optional).
#
# Returns:
# A `Tensor` of the same shape as `labels` and of the same type as
# `logits`
# with the softmax cross entropy loss.
#
# Raises:
# ValueError: If logits are scalars (need to have rank >= 1) or if
# the rank
# of the labels is not equal to the rank of the labels minus one.
# ```

# In[ ]:

with tf.variable_scope('loss'):
    # Compute mean cross entropy loss for each output.
    Y_true_flat = tf.reshape(tf.concat(1, Y), [-1])
    # logits are [batch_size x timesteps, n_chars] and
    # Y_true_flat are [batch_size x timesteps]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y_true_flat)
    # Compute the mean over our `batch_size` x `timesteps` number of observations
    mean_loss = tf.reduce_mean(loss)


# Finally, we can create an optimizer in much the same way as we've
# done with every other network. Except, we will also "clip" the
# gradients of every trainable parameter. This is a hacky way to
# ensure that the gradients do not grow too large (the literature
# calls this the "exploding gradient problem"). However, note that
# the LSTM is built to help ensure this does not happen by allowing
# the gradient to be "gated". To learn more about this, please
# consider reading the following material:
#
# http://www.felixgers.de/papers/phd.pdf
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/

# In[ ]:

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    gradients = []
    clip = tf.constant(5.0, name="clip")
    for grad, var in optimizer.compute_gradients(mean_loss):
        gradients.append((tf.clip_by_value(grad, -clip, clip), var))
    updates = optimizer.apply_gradients(gradients)


# Let's take a look at the graph:

# In[ ]:

nb_utils.show_graph(tf.get_default_graph().as_graph_def())


# Below is the rest of code we'll need to train the network. I do not
# recommend running this inside Jupyter Notebook for the entire
# length of the training because the network can take 1-2 days at
# least to train, and your browser may very likely complain. Instead,
# you should write a python script containing the necessary bits of
# code and run it using the Terminal. We didn't go over how to do
# this, so I'll leave it for you as an exercise. The next part of
# this notebook will have you load a pre-trained network.

# In[ ]:

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    cursor = 0
    it_i = 0
    while it_i < 500:
        Xs, Ys = [], []
        for batch_i in range(batch_size):
            if (cursor + sequence_length) >= len(txt) - sequence_length - 1:
                cursor = 0
            Xs.append([encoder[ch]
                       for ch in txt[cursor:cursor + sequence_length]])
            Ys.append([encoder[ch]
                       for ch in txt[cursor + 1: cursor + sequence_length + 1]])

            cursor = (cursor + sequence_length)
        Xs = np.array(Xs).astype(np.int32)
        Ys = np.array(Ys).astype(np.int32)

        loss_val, _ = sess.run([mean_loss, updates],
                               feed_dict={X: Xs, Y: Ys})
        if it_i % 100 == 0:
            print(it_i, loss_val)

        if it_i % 500 == 0:
            p = sess.run(probs, feed_dict={X: np.array(Xs[-1])[np.newaxis]})
            ps = [np.random.choice(range(n_chars), p=p_i.ravel())
                  for p_i in p]
            p = [np.argmax(p_i) for p_i in p]
            if isinstance(txt[0], str):
                print('original:', "".join(
                    [decoder[ch] for ch in Xs[-1]]))
                print('synth(samp):', "".join(
                    [decoder[ch] for ch in ps]))
                print('synth(amax):', "".join(
                    [decoder[ch] for ch in p]))
            else:
                print([decoder[ch] for ch in ps])

        it_i += 1


# <a name="part-5---pretrained-char-rnn-of-donald-trump"></a>
# # Part 5 - Pretrained Char-RNN of Donald Trump
#
# Rather than stick around to let a model train, let's now explore
# one I've trained for you Donald Trump. If you've trained your own
# model on your own text corpus then great! You should be able to use
# that in place of the one I've provided and still continue with the
# rest of the notebook.
#
# For the Donald Trump corpus, there are a lot of video transcripts
# that you can find online. I've searched for a few of these, put
# them in a giant text file, made everything lowercase, and removed
# any extraneous letters/symbols to help reduce the vocabulary (not
# that it's not very large to begin with, ha).
#
# I used the code exactly as above to train on the text I gathered
# and left it to train for about 2 days. The only modification is
# that I also used "dropout" which you can see in the libs/charrnn.py
# file. Let's explore it now and we'll see how we can play with
# "sampling" the model to generate new phrases, and how to "prime"
# the model (a psychological term referring to when someone is
# exposed to something shortly before another event).
#
# First, let's clean up any existing graph:

# In[ ]:

tf.reset_default_graph()


# <a name="getting-the-trump-data"></a>
# ## Getting the Trump Data
#
# Now let's load the text. This is included in the repo or can be
# downloaded from:

# In[ ]:

with open('trump.txt', 'r') as fp:
    txt = fp.read()


# Let's take a look at what's going on in here:

# In[ ]:

txt[:100]


# <a name="basic-text-analysis"></a>
# ## Basic Text Analysis
#
# We can do some basic data analysis to get a sense of what kind of
# vocabulary we're working with. It's really important to look at
# your data in as many ways as possible. This helps ensure there
# isn't anything unexpected going on. Let's find every unique word he
# uses:

# In[ ]:

words = set(txt.split(' '))


# In[ ]:

words


# Now let's count their occurrences:

# In[ ]:

counts = {word_i: 0 for word_i in words}
for word_i in txt.split(' '):
    counts[word_i] += 1
counts


# We can sort this like so:

# In[ ]:

[(word_i, counts[word_i]) for word_i in sorted(counts, key=counts.get, reverse=True)]


# As we should expect, "the" is the most common word, as it is in the
# English language:
# https://en.wikipedia.org/wiki/Most_common_words_in_English
#
# <a name="loading-the-pre-trained-trump-model"></a>
# ## Loading the Pre-trained Trump Model
#
# Let's load the pretrained model. Rather than provide a tfmodel
# export, I've provided the checkpoint so you can also experiment
# with training it more if you wish. We'll rebuild the graph using
# the `charrnn` module in the `libs` directory:

# In[ ]:

from libs import charrnn


# Let's get the checkpoint and build the model then restore the
# variables from the checkpoint. The only parameters of consequence
# are `n_layers` and `n_cells` which define the total size and layout
# of the model. The rest are flexible. We'll set the `batch_size` and
# `sequence_length` to 1, meaning we can feed in a single character
# at a time only, and get back 1 character denoting the very next
# character's prediction.

# In[ ]:

ckpt_name = 'trump.ckpt'
g = tf.Graph()
n_layers = 3
n_cells = 512
with tf.Session(graph=g) as sess:
    model = charrnn.build_model(txt=txt,
                                batch_size=1,
                                sequence_length=1,
                                n_layers=n_layers,
                                n_cells=n_cells,
                                gradient_clip=10.0)
    saver = tf.train.Saver()
    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print("Model restored.")


# Let's now take a look at the model:

# In[ ]:

nb_utils.show_graph(g.as_graph_def())


# In[ ]:

n_iterations = 100


# <a name="inference-keeping-track-of-the-state"></a>
# ## Inference: Keeping Track of the State
#
# Now recall from Part 4 when we created our LSTM network, we had an
# `initial_state` variable which would set the LSTM's `c` and `h`
# state vectors, as well as the final output state which was the
# output of the `c` and `h` state vectors after having passed through
# the network. When we input to the network some letter, say 'n', we
# can set the `initial_state` to zeros, but then after having input
# the letter `n`, we'll have as output a new state vector for `c` and
# `h`. On the next letter, we'll then want to set the `initial_state`
# to this new state, and set the input to the previous letter's
# output. That is how we ensure the network keeps track of time and
# knows what has happened in the past, and let it continually
# generate.

# In[ ]:

curr_states = None
g = tf.Graph()
with tf.Session(graph=g) as sess:
    model = charrnn.build_model(txt=txt,
                                batch_size=1,
                                sequence_length=1,
                                n_layers=n_layers,
                                n_cells=n_cells,
                                gradient_clip=10.0)
    saver = tf.train.Saver()
    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print("Model restored.")
        
    # Get every tf.Tensor for the initial state
    init_states = []
    for s_i in model['initial_state']:
        init_states.append(s_i.c)
        init_states.append(s_i.h)
        
    # Similarly, for every state after inference
    final_states = []
    for s_i in model['final_state']:
        final_states.append(s_i.c)
        final_states.append(s_i.h)

    # Let's start with the letter 't' and see what comes out:
    synth = [[encoder[' ']]]
    for i in range(n_iterations):

        # We'll create a feed_dict parameter which includes what to
        # input to the network, model['X'], as well as setting
        # dropout to 1.0, meaning no dropout.
        feed_dict = {model['X']: [synth[-1]],
                     model['keep_prob']: 1.0}
        
        # Now we'll check if we currently have a state as a result
        # of a previous inference, and if so, add to our feed_dict
        # parameter the mapping of the init_state to the previous
        # output state stored in "curr_states".
        if curr_states:
            feed_dict.update(
                {init_state_i: curr_state_i
                 for (init_state_i, curr_state_i) in
                     zip(init_states, curr_states)})
            
        # Now we can infer and see what letter we get
        p = sess.run(model['probs'], feed_dict=feed_dict)[0]
        
        # And make sure we also keep track of the new state
        curr_states = sess.run(final_states, feed_dict=feed_dict)
        
        # Find the most likely character
        p = np.argmax(p)
        
        # Append to string
        synth.append([p])
        
        # Print out the decoded letter
        print(model['decoder'][p], end='')
        sys.stdout.flush()


# <a name="probabilistic-sampling"></a>
# ## Probabilistic Sampling
#
# Run the above cell a couple times. What you should find is that it
# is deterministic. We always pick *the* most likely character. But
# we can do something else which will make things less deterministic
# and a bit more interesting: we can sample from our probabilistic
# measure from our softmax layer. This means if we have the letter
# 'a' as 0.4, and the letter 'o' as 0.2, we'll have a 40% chance of
# picking the letter 'a', and 20% chance of picking the letter 'o',
# rather than simply always picking the letter 'a' since it is the
# most probable.

# In[ ]:

curr_states = None
g = tf.Graph()
with tf.Session(graph=g) as sess:
    model = charrnn.build_model(txt=txt,
                                batch_size=1,
                                sequence_length=1,
                                n_layers=n_layers,
                                n_cells=n_cells,
                                gradient_clip=10.0)
    saver = tf.train.Saver()
    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print("Model restored.")
        
    # Get every tf.Tensor for the initial state
    init_states = []
    for s_i in model['initial_state']:
        init_states.append(s_i.c)
        init_states.append(s_i.h)
        
    # Similarly, for every state after inference
    final_states = []
    for s_i in model['final_state']:
        final_states.append(s_i.c)
        final_states.append(s_i.h)

    # Let's start with the letter 't' and see what comes out:
    synth = [[encoder[' ']]]
    for i in range(n_iterations):

        # We'll create a feed_dict parameter which includes what to
        # input to the network, model['X'], as well as setting
        # dropout to 1.0, meaning no dropout.
        feed_dict = {model['X']: [synth[-1]],
                     model['keep_prob']: 1.0}
        
        # Now we'll check if we currently have a state as a result
        # of a previous inference, and if so, add to our feed_dict
        # parameter the mapping of the init_state to the previous
        # output state stored in "curr_states".
        if curr_states:
            feed_dict.update(
                {init_state_i: curr_state_i
                 for (init_state_i, curr_state_i) in
                     zip(init_states, curr_states)})
            
        # Now we can infer and see what letter we get
        p = sess.run(model['probs'], feed_dict=feed_dict)[0]
        
        # And make sure we also keep track of the new state
        curr_states = sess.run(final_states, feed_dict=feed_dict)
        
        # Now instead of finding the most likely character,
        # we'll sample with the probabilities of each letter
        p = p.astype(np.float64)
        p = np.random.multinomial(1, p.ravel() / p.sum())
        p = np.argmax(p)
        
        # Append to string
        synth.append([p])
        
        # Print out the decoded letter
        print(model['decoder'][p], end='')
        sys.stdout.flush()


# <a name="inference-temperature"></a>
# ## Inference: Temperature
#
# When performing probabilistic sampling, we can also use a parameter
# known as temperature which comes from simulated annealing. The
# basic idea is that as the temperature is high and very hot, we have
# a lot more free energy to use to jump around more, and as we cool
# down, we have less energy and then become more deterministic. We
# can use temperature by scaling our log probabilities like so:

# In[ ]:

temperature = 0.5
curr_states = None
g = tf.Graph()
with tf.Session(graph=g) as sess:
    model = charrnn.build_model(txt=txt,
                                batch_size=1,
                                sequence_length=1,
                                n_layers=n_layers,
                                n_cells=n_cells,
                                gradient_clip=10.0)
    saver = tf.train.Saver()
    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print("Model restored.")
        
    # Get every tf.Tensor for the initial state
    init_states = []
    for s_i in model['initial_state']:
        init_states.append(s_i.c)
        init_states.append(s_i.h)
        
    # Similarly, for every state after inference
    final_states = []
    for s_i in model['final_state']:
        final_states.append(s_i.c)
        final_states.append(s_i.h)

    # Let's start with the letter 't' and see what comes out:
    synth = [[encoder[' ']]]
    for i in range(n_iterations):

        # We'll create a feed_dict parameter which includes what to
        # input to the network, model['X'], as well as setting
        # dropout to 1.0, meaning no dropout.
        feed_dict = {model['X']: [synth[-1]],
                     model['keep_prob']: 1.0}
        
        # Now we'll check if we currently have a state as a result
        # of a previous inference, and if so, add to our feed_dict
        # parameter the mapping of the init_state to the previous
        # output state stored in "curr_states".
        if curr_states:
            feed_dict.update(
                {init_state_i: curr_state_i
                 for (init_state_i, curr_state_i) in
                     zip(init_states, curr_states)})
            
        # Now we can infer and see what letter we get
        p = sess.run(model['probs'], feed_dict=feed_dict)[0]
        
        # And make sure we also keep track of the new state
        curr_states = sess.run(final_states, feed_dict=feed_dict)
        
        # Now instead of finding the most likely character,
        # we'll sample with the probabilities of each letter
        p = p.astype(np.float64)
        p = np.log(p) / temperature
        p = np.exp(p) / np.sum(np.exp(p))
        p = np.random.multinomial(1, p.ravel() / p.sum())
        p = np.argmax(p)
        
        # Append to string
        synth.append([p])
        
        # Print out the decoded letter
        print(model['decoder'][p], end='')
        sys.stdout.flush()


# <a name="inference-priming"></a>
# ## Inference: Priming
#
# Let's now work on "priming" the model with some text, and see what
# kind of state it is in and leave it to synthesize from there. We'll
# do more or less what we did before, but feed in our own text
# instead of the last letter of the synthesis from the model.

# In[ ]:

prime = "obama"
temperature = 1.0
curr_states = None
n_iterations = 500
g = tf.Graph()
with tf.Session(graph=g) as sess:
    model = charrnn.build_model(txt=txt,
                                batch_size=1,
                                sequence_length=1,
                                n_layers=n_layers,
                                n_cells=n_cells,
                                gradient_clip=10.0)
    saver = tf.train.Saver()
    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print("Model restored.")
        
    # Get every tf.Tensor for the initial state
    init_states = []
    for s_i in model['initial_state']:
        init_states.append(s_i.c)
        init_states.append(s_i.h)
        
    # Similarly, for every state after inference
    final_states = []
    for s_i in model['final_state']:
        final_states.append(s_i.c)
        final_states.append(s_i.h)

    # Now we'll keep track of the state as we feed it one
    # letter at a time.
    curr_states = None
    for ch in prime:
        feed_dict = {model['X']: [[model['encoder'][ch]]],
                     model['keep_prob']: 1.0}
        if curr_states:
            feed_dict.update(
                {init_state_i: curr_state_i
                 for (init_state_i, curr_state_i) in
                     zip(init_states, curr_states)})
        
        # Now we can infer and see what letter we get
        p = sess.run(model['probs'], feed_dict=feed_dict)[0]
        p = p.astype(np.float64)
        p = np.log(p) / temperature
        p = np.exp(p) / np.sum(np.exp(p))
        p = np.random.multinomial(1, p.ravel() / p.sum())
        p = np.argmax(p)
        
        # And make sure we also keep track of the new state
        curr_states = sess.run(final_states, feed_dict=feed_dict)
        
    # Now we're ready to do what we were doing before but with the
    # last predicted output stored in `p`, and the current state of
    # the model.
    synth = [[p]]
    print(prime + model['decoder'][p], end='')
    for i in range(n_iterations):

        # Input to the network
        feed_dict = {model['X']: [synth[-1]],
                     model['keep_prob']: 1.0}
        
        # Also feed our current state
        feed_dict.update(
            {init_state_i: curr_state_i
             for (init_state_i, curr_state_i) in
                 zip(init_states, curr_states)})
            
        # Inference
        p = sess.run(model['probs'], feed_dict=feed_dict)[0]
        
        # Keep track of the new state
        curr_states = sess.run(final_states, feed_dict=feed_dict)
        
        # Sample
        p = p.astype(np.float64)
        p = np.log(p) / temperature
        p = np.exp(p) / np.sum(np.exp(p))
        p = np.random.multinomial(1, p.ravel() / p.sum())
        p = np.argmax(p)
        
        # Append to string
        synth.append([p])
        
        # Print out the decoded letter
        print(model['decoder'][p], end='')
        sys.stdout.flush()


# <a name="assignment-submission"></a>
# # Assignment Submission
# After you've completed both notebooks, create a zip file of the
# current directory using the code below. This code will make sure
# you have included this completed ipython notebook and the following
# files named exactly as:
#
# session-5/
# session-5-part-1.ipynb
# session-5-part-2.ipynb
# vaegan.gif

# You'll then submit this zip file for your third assignment on
# Kadenze for "Assignment 5: Generative Adversarial Networks and
# Recurrent Neural Networks"! If you have any questions, remember to
# reach out on the forums and connect with your peers or with me.
#
# To get assessed, you'll need to be a premium student! This will
# allow you to build an online portfolio of all of your work and
# receive grades. If you aren't already enrolled as a student,
# register now at http://www.kadenze.com/ and join the #CADL
# community to see what your peers are doing!
# https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info
#
# Also, if you share any of the GIFs on
# Facebook/Twitter/Instagram/etc..., be sure to use the #CADL hashtag
# so that other students can find your work!

# In[ ]:

utils.build_submission('session-5.zip',
                       ('vaegan.gif',
                        'session-5-part-1.ipynb',
                        'session-5-part-2.ipynb'))

