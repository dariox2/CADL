
#
# Training a Network w/ Tensorflow
#

# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda '
          'and then restart `jupyter notebook`:\n' \
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
except ImportError:
    print('You are missing some packages! ' \
          'We will try installing them before continuing!')
    ###!pip install "numpy>=1.11.0" "matplotlib>=1.5.1" "scikit-image>=0.11.3" "scikit-learn>=0.17" "scipy>=0.17.0"
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    print('Done!')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")
    print("https://github.com/pkmital/CADL#installation-preliminaries")

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif
try:
    from libs import utils, gif
    import IPython.display as ipyd
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this assignment unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")

# We'll tell matplotlib to inline any drawn figures like so:
###%matplotlib inline
plt.style.use('ggplot')

#--2
# Bit of formatting because I don't like the default inline code style:

#
# Part One - Fully Connected Network
#

xs = np.linspace(-6, 6, 100)
plt.plot(xs, np.maximum(xs, 0), label='relu')
plt.plot(xs, 1 / (1 + np.exp(-xs)), label='sigmoid')
plt.plot(xs, np.tanh(xs), label='tanh')
plt.xlabel('Input')
plt.xlim([-6, 6])
plt.ylabel('Output')
plt.ylim([-1.5, 1.5])
plt.title('Common Activation Functions/Nonlinearities')
plt.legend(loc='lower right')

#dja
plt.ion()
#plt.show()
#plt.pause(2)
plt.close()

#
# Code
#


# TODO! COMPLETE THIS SECTION!
# Create a placeholder with None x 2 dimensions of dtype tf.float32, and name it "X":
X = tf.placeholder(tf.float32, shape=(None, 2), name="X")


# TODO! COMPLETE THIS SECTION!
W = tf.get_variable("W", (2,20), initializer=tf.random_normal_initializer())
h = tf.matmul(X, W)


# TODO! COMPLETE THIS SECTION!
b = tf.get_variable("b", 20, initializer=tf.constant_initializer(1.0))
h = tf.nn.bias_add(h, b)


# TODO! COMPLETE THIS SECTION!
h = tf.nn.relu(h + b)


#
# Variable Scopes
#

h, W = utils.linear(
    x=X, n_output=20, name='linear', activation=tf.nn.relu)

#
# Part Two - Image Painting Network
#

#
# Preparing the Data
#


# TODO! COMPLETE THIS SECTION!
# First load an image
import matplotlib.pyplot as plt
#img = plt.imread("mypictures/2000px-Tux.svg.png")
img = plt.imread("mypictures/tux-small.jpg")
#
# Be careful with the size of your image.
# Try a fairly small image to begin with,
# then come back here and try larger sizes.
img = imresize(img, (100, 100))
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.title("(preparing the data)")
plt.show()
plt.pause(5)
plt.close()
#
# Make sure you save this image as "reference.png"
# and include it in your zipped submission file
# so we can tell what image you are trying to paint!
plt.imsave(fname='reference_batch.png', arr=img)


print(img.shape)

def split_image(img):
    # We'll first collect all the positions in the image in our list, xs
    xs = []

    # And the corresponding colors for each of these positions
    ys = []

    # Now loop over the image
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            # And store the inputs
            xs.append([row_i, col_i])
            # And outputs that the network needs to learn to predict
            ys.append(img[row_i, col_i])

    # we'll convert our lists to arrays
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

xs, ys = split_image(img)
# and print the shapes
xs.shape, ys.shape


# TODO! COMPLETE THIS SECTION!
# Normalize the input (xs) using its mean and standard deviation
xs = (xs - np.mean(xs)) / np.std(xs)
#
# Just to make sure you have normalized it correctly:
print(np.min(xs), np.max(xs))
assert(np.min(xs) > -3.0 and np.max(xs) < 3.0)


print(np.min(ys), np.max(ys))

ys = ys / 255.0
print(np.min(ys), np.max(ys))

plt.imshow(ys.reshape(img.shape))
plt.title("(reshape)")
#plt.show()
#plt.pause(2)
plt.close()


# TODO! COMPLETE THIS SECTION!
# Let's reset the graph:
tf.reset_default_graph()
#
# Create a placeholder of None x 2 dimensions and dtype tf.float32
# This will be the input to the network which takes the row/col
X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
#
# Create the placeholder, Y, with 3 output dimensions instead of 2.
# This will be the output of the network, the R, G, B values.
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')


# TODO! COMPLETE THIS SECTION!
# We'll create 6 hidden layers.  Let's create a variable
# to say how many neurons we want for each of the layers
# (try 20 to begin with, then explore other values)
n_neurons = [2, 20,20,20,20,20,20, 3]
#
# Create the first linear + nonlinear layer which will
# take the 2 input neurons and fully connects it to 20 neurons.
# Use the `utils.linear` function to do this just like before,
# but also remember to give names for each layer, such as
# "1", "2", ... "5", or "layer1", "layer2", ... "layer6".
h1, W1 = utils.linear(X, 20, activation=None, name='Lay1')
#
# Create another one:
h2, W2 = utils.linear(h1, 20, activation=None, name='Lay2')
#
# and four more (or replace all of this with a loop if you can!):
h3, W3 = utils.linear(h2, 20, activation=None, name='Lay3')
h4, W4 = utils.linear(h3, 20, activation=None, name='Lay4')
h5, W5 = utils.linear(h4, 20, activation=None, name='Lay5')
h6, W6 = utils.linear(h5, 20, activation=None, name='Lay6')
#
# Now, make one last layer to make sure your network has 3 outputs:
Y_pred, W7 = utils.linear(h6, 3, activation=None, name='pred')


assert(X.get_shape().as_list() == [None, 2])
assert(Y_pred.get_shape().as_list() == [None, 3])
assert(Y.get_shape().as_list() == [None, 3])

#
# Cost Function
#
error = np.linspace(0.0, 128.0**2, 100)
loss = error**2.0
plt.plot(error, loss)
plt.xlabel('error')
plt.ylabel('loss')
plt.title("(cost function)")
#plt.show()
#plt.pause(2)
plt.close()


error = np.linspace(0.0, 1.0, 100)
plt.plot(error, error**2, label='l_2 loss')
plt.plot(error, np.abs(error), label='l_1 loss')
plt.xlabel('error')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.title("(l1 vs l2 loss)")
#plt.show()
#plt.pause(2)
plt.close()


#-- hasta aqui todo ok


# TODO! COMPLETE THIS SECTION!
# first compute the error, the inner part of the summation.
# This should be the l1-norm or l2-norm of the distance
# between each color channel.
error = tf.abs(tf.sub(Y*Y, Y_pred*Y_pred))
assert(error.get_shape().as_list() == [None, 3])


# TODO! COMPLETE THIS SECTION!
# Now sum the error for each feature in Y. 
# If Y is [Batch, Features], the sum should be [Batch]:
sum_error = tf.reduce_sum(error, 1)
assert(sum_error.get_shape().as_list() == [None])


# TODO! COMPLETE THIS SECTION!
# Finally, compute the cost, as the mean error of the batch.
# This should be a single value.
cost = tf.reduce_mean(sum_error)
assert(cost.get_shape().as_list() == [])


# TODO! COMPLETE THIS SECTION!
# Refer to the help for the function
optimizer =tf.train.AdamOptimizer(0.001).minimize(cost)
#
# Create parameters for the number of iterations to run for (< 100)
n_iterations = 500
#
# And how much data is in each minibatch (< 500)
batch_size = 200
#
# Then create a session
sess = tf.Session()


# Initialize all your variables and run the operation with your session
sess.run(tf.initialize_all_variables())

# Optimize over a few iterations, each time following the gradient
# a little at a time
imgs = []
costs = []
gif_step = n_iterations // 10
step_i = 0

for it_i in range(n_iterations):
    
    # Get a random sampling of the dataset
    idxs = np.random.permutation(range(len(xs)))
    
    # The number of batches we have to iterate over
    n_batches = len(idxs) // batch_size
    
    # Now iterate over our stochastic minibatches:
    for batch_i in range(n_batches):
         
        # Get just minibatch amount of data
        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

        # And optimize, also returning the cost so we can monitor
        # how our optimization is doing.
        training_cost = sess.run(
            [cost, optimizer],
            feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]

    # Also, every 20 iterations, we'll draw the prediction of our
    # input xs, which should try to recreate our image!
    if (it_i + 1) % gif_step == 0:
        costs.append(training_cost / n_batches)
        ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
        img = np.clip(ys_pred.reshape(img.shape), 0, 1)
        imgs.append(img)
        # Plot the cost over time
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(costs)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Cost')
        ax[1].imshow(img)
        fig.suptitle('Iteration {}'.format(it_i))
        plt.show()
        plt.pause(1)
        plt.close()

# Save the images as a GIF
_ = gif.build_gif(imgs, saveto='single_batch.gif', show_gif=False)


#ipyd.Image(url='single_batch.gif?{}'.format(np.random.rand()), height=500, width=500)


# eop


