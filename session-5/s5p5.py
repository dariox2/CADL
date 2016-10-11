

# Session 5, part 5 (notebook part 2)

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

def wait(n):
    #plt.pause(n)
    plt.pause(3)
    #input("(press enter)")


##
## Part 5 - Pretrained Char-RNN of Donald Trump
##


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

tf.reset_default_graph()


#
# Getting the Trump Data
#

# Now let's load the text. This is included in the repo or can be
# downloaded from:

with open('trump.txt', 'r') as fp:
    txt = fp.read()


# Let's take a look at what's going on in here:

# In[ ]:

txt[:100]


#
# Basic Text Analysis
#

# We can do some basic data analysis to get a sense of what kind of
# vocabulary we're working with. It's really important to look at
# your data in as many ways as possible. This helps ensure there
# isn't anything unexpected going on. Let's find every unique word he
# uses:

words = set(txt.split(' '))


print("words: ", words)


# Now let's count their occurrences:

counts = {word_i: 0 for word_i in words}
for word_i in txt.split(' '):
    counts[word_i] += 1

print("counts: ", counts)


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

# nb_utils.show_graph(g.as_graph_def())


n_iterations = 100


#
# Inference: Keeping Track of the State
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


#
# Probabilistic Sampling
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


#
# Inference: Temperature
#

# When performing probabilistic sampling, we can also use a parameter
# known as temperature which comes from simulated annealing. The
# basic idea is that as the temperature is high and very hot, we have
# a lot more free energy to use to jump around more, and as we cool
# down, we have less energy and then become more deterministic. We
# can use temperature by scaling our log probabilities like so:


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


#
# Inference: Priming
#

# Let's now work on "priming" the model with some text, and see what
# kind of state it is in and leave it to synthesize from there. We'll
# do more or less what we did before, but feed in our own text
# instead of the last letter of the synthesis from the model.

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


# eop 

