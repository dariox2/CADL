
#
# Session 3, part 3
#
# Audio classification w/convolutional network
#

import sys

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import IPython.display as ipyd

print("Loading tensorflow...")
import tensorflow as tf

from libs import utils, gif, datasets, dataset_utils, vae, dft
from libs.utils import montage_filters

plt.style.use('bmh')

import datetime

# dja
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
plt.figure(figsize=(5, 5))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")

#
# Preparing the data
#

#dst = 'gtzan_music_speech'
dst="../../gtzan_music_speech"
if not os.path.exists(dst):
    dataset_utils.gtzan_music_speech_download(dst)


# Get the full path to the directory
music_dir = os.path.join(os.path.join(dst, 'music_speech'), 'music_wav')

# Now use list comprehension to combine the path of the directory with any wave files
music = [os.path.join(music_dir, file_i)
         for file_i in os.listdir(music_dir)
         if file_i.endswith('.wav')]

# Similarly, for the speech folder:
speech_dir = os.path.join(os.path.join(dst, 'music_speech'), 'speech_wav')
speech = [os.path.join(speech_dir, file_i)
          for file_i in os.listdir(speech_dir)
          if file_i.endswith('.wav')]

# dja - debug with fewer files
print("len(music): ", len(music), "  len(speech): ", len(speech))
music=music[0:15]
speech=speech[0:15]

# Let's see all the file names
print("new length: ", len(music), len(speech))


file_i = music[0]
s = utils.load_audio(file_i)
plt.title("sample 0")
plt.plot(s)

plt.pause(3)

# Parameters for our dft transform.  Sorry we can't go into the
# details of this in this course.  Please look into DSP texts or the
# course by Perry Cook linked above if you are unfamiliar with this.
fft_size = 512
hop_size = 256

re, im = dft.dft_np(s, hop_size=256, fft_size=512)
mag, phs = dft.ztoc(re, im)
print("mag.shape: ", mag.shape)
plt.title("magnitude")
plt.imshow(mag)
plt.pause(3)
plt.title("phase")
plt.imshow(phs)
plt.pause(3)


#plt.figure(figsize=(10, 4))
plt.title("log(mag)")
plt.imshow(np.log(mag.T))
plt.xlabel('Time')
plt.ylabel('Frequency Bin')
plt.pause(3)

# The sample rate from our audio is 22050 Hz.
sr = 22050

# We can calculate how many hops there are in a second
# which will tell us how many frames of magnitudes
# we have per second
n_frames_per_second = sr // hop_size

# We want 500 milliseconds of audio in our window
n_frames = n_frames_per_second // 2

# And we'll move our window by 250 ms at a time
frame_hops = n_frames_per_second // 4

# We'll therefore have this many sliding windows:
#n_hops = (len(mag) - n_frames) // frame_hops # WRONG!
n_hops = len(mag) // frame_hops - 1 # dja

print("frames/s: ", n_frames_per_second)
print("# frames: ", n_frames)
print("frame hops: ", frame_hops)
print("# hops: ", n_hops)

Xs = []
ys = []
for hop_i in range(n_hops):
    #print("hop: ", hop_i)
    # Creating our sliding window
    frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]
    
    # Store them with a new 3rd axis and as a logarithmic scale
    # We'll ensure that we aren't taking a log of 0 just by adding
    # a small value, also known as epsilon.
    Xs.append(np.log(np.abs(frames[..., np.newaxis]) + 1e-10))
    
    # And then store the label 
    ys.append(0)


plt.imshow(Xs[0][..., 0])
plt.title('label:{}'.format(ys[1]) + " (1st window)")
plt.pause(3)

plt.imshow(Xs[1][..., 0])
plt.title('label:{}'.format(ys[1]) + " (2nd window)")
plt.pause(3)

# TODO
# Store every magnitude frame and its label of being music: 0 or speech: 1
Xs, ys = [], []

# Let's start with the music files
for idx, fn in enumerate(music):
    print("music #",idx, " ", fn)
    # Load the ith file:
    s = utils.load_audio(fn)
    
    # Now take the dft of it (take a DSP course!):
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)
    
    # And convert the complex representation to magnitudes/phases (take a DSP course!):
    mag, phs = dft.ztoc(re, im)
    
    # This is how many sliding windows we have:
    #n_hops = (len(mag) - n_frames) // frame_hops # WRONG!
    n_hops = len(mag) // frame_hops - 1 # dja
    
    # Let's extract them all:
    for hop_i in range(n_hops):
        
        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]
        
        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)
        
        # And store it:
        Xs.append(this_X)
        
        # And be sure that we store the correct label of this observation:
        ys.append(0)
        
# Now do the same thing with speech (TODO)!
for idx, fn in enumerate(music):
    print("speech #",idx, " ", fn)
    # Load the ith file:
    s = utils.load_audio(fn)
    
    # Now take the dft of it (take a DSP course!):
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)
    
    # And convert the complex representation to magnitudes/phases (take a DSP course!):
    mag, phs = dft.ztoc(re, im)
    
    # This is how many sliding windows we have:
    #n_hops = (len(mag) - n_frames) // frame_hops # WRONG!
    n_hops = len(mag) // frame_hops - 1 # dja

    # Let's extract them all:
    for hop_i in range(n_hops):
        
        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]
        
        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)
        
        # And store it:
        Xs.append(this_X)
        
        # Make sure we use the right label (TODO!)!
        ys.append(1)
        
# Convert them to an array:
print("Converting to array...")
Xs = np.array(Xs)
ys = np.array(ys)

print(Xs.shape, ys.shape)

# Just to make sure you've done it right.  If you've changed any of the
# parameters of the dft/hop size, then this will fail.  If that's what you
# wanted to do, then don't worry about this assertion.
#assert(Xs.shape == (15360, 43, 256, 1) and ys.shape == (15360,))
assert(Xs.shape == ((len(music)+len(speech))*n_hops, n_frames, hop_size, 1) \
       and ys.shape == ((len(music)+len(speech))*n_hops,))


plt.imshow(Xs[0][..., 0])
plt.title('label:{}'.format(ys[0])+ "1st magnitude matrix")
plt.pause(5)

n_observations, n_height, n_width, n_channels = Xs.shape


# TODO
print("Creating dataset object...")
ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], one_hot=True)


print("obtaining minibatch...")
Xs_i, ys_i = next(ds.train.next_batch())

# Notice the shape this returns.  This will become the shape of our input and output of the network:
print("batch shapes, Xs: ", Xs_i.shape, " ys: ", ys_i.shape)

assert(ys_i.shape == (batch_size, 2))


plt.imshow(Xs_i[0, :, :, 0])
plt.title('label:{}'.format(ys_i[0])+" (1st random batch)")
plt.pause(3)


plt.imshow(Xs_i[1, :, :, 0])
plt.title('label:{}'.format(ys_i[1])+" (2nd random batch)")
plt.pause(3)


#
# Creating the Network
#

# TODO
tf.reset_default_graph()

# Create the input to the network.  This is a 4-dimensional tensor!
# Recall that we are using sliding windows of our magnitudes (TODO):
X = tf.placeholder(name='X', shape=[None, n_frames, hop_size, 1], dtype=tf.float32)

# Create the output to the network.  This is our one hot encoding of 2 possible values (TODO)!
Y = tf.placeholder(name='Y', shape=[None, 2], dtype=tf.float32)


# TODO:  Explore different numbers of layers, and sizes of the network
n_filters = [9, 9, 9, 9]

# Now let's loop over our n_filters and create the deep convolutional neural network
H = X
for layer_i, n_filters_i in enumerate(n_filters):
    
    # Let's use the helper function to create our connection to the next layer:
    # TODO: explore changing the parameters here:
    H, W = utils.conv2d(
        H, n_filters_i, k_h=3, k_w=3, d_h=2, d_w=2,
        name=str(layer_i))
    
    # And use a nonlinearity
    # TODO: explore changing the activation here:
    H = tf.nn.relu(H)
    
    # Just to check what's happening:
    print("layer ", layer_i, " shape: ", H.get_shape().as_list())


# TODO
# Connect the last convolutional layer to a fully connected network (TODO)!
fc, W = utils.linear(x=H, n_output=ys_i.shape[0], activation=tf.nn.relu, name="layer_last_full")

# And another fully connceted network, now with just 2 outputs, the number of outputs that our
# one hot encoding has (TODO)!
Y_pred, W = utils.linear(x=fc, n_output=2, activation=tf.nn.softmax, name="layer_last_out")


loss = utils.binary_cross_entropy(Y_pred, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))


# TODO
predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# TODO
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



# Explore these parameters: (TODO)
n_epochs = 10
batch_size = 100#200

# Create a session and init!
sess = tf.Session()
sess.run(tf.initialize_all_variables())

print("Training...")
t1 = datetime.datetime.now()

# Now iterate over our dataset n_epoch times
for epoch_i in range(n_epochs):
    print('Epoch: ', epoch_i)
    
    # Train
    this_accuracy = 0
    its = 0
    
    # Do our mini batches:
    for Xs_i, ys_i in ds.train.next_batch(batch_size):
        # Note here: we are running the optimizer so
        # that the network parameters train!
        this_accuracy += sess.run([accuracy, optimizer], feed_dict={
                X:Xs_i, Y:ys_i})[0]
        its += 1
        #print(this_accuracy / its)
    print('Training accuracy: ', this_accuracy / its)
    
    # Validation (see how the network does on unseen data).
    this_accuracy = 0
    its = 0
    
    # Do our mini batches:
    for Xs_i, ys_i in ds.valid.next_batch(batch_size):
        # Note here: we are NOT running the optimizer!
        # we only measure the accuracy!
        this_accuracy += sess.run(accuracy, feed_dict={
                X:Xs_i, Y:ys_i})
        its += 1
    print('Validation accuracy: ', this_accuracy / its)

t2 = datetime.datetime.now()
delta = t2 - t1
print("             Total training time: ", delta.total_seconds())

#print("graph:")
#g = tf.get_default_graph()
#for op in g.get_operations():
#  print(op.name)

# TODO
g = tf.get_default_graph()
W = W = sess.run(g.get_tensor_by_name('0/W:0'))

assert(W.dtype == np.float32)
m = montage_filters(W)
#plt.figure(figsize=(5, 5))
plt.title("first layer weights")
plt.imshow(m)
plt.imsave(arr=m, fname='audio.png')
plt.pause(3)

g = tf.get_default_graph()
for layer_i in range(len(n_filters)):
    W = sess.run(g.get_tensor_by_name('{}/W:0'.format(layer_i)))
    #plt.figure(figsize=(5, 5))
    plt.imshow(montage_filters(W))
    plt.title('Layer {}\'s Learned Convolution Kernels'.format(layer_i))
    plt.pause(1)

plt.pause(3)

#eop
