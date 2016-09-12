
#
# Session 3, part 1
#
# Encoder/decoder
#
# Biswal home, Total training time:  321.489693
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

#plt.style.use('ggplot')
plt.style.use('bmh')

import datetime

# dja
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
plt.figure(figsize=(3, 3))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")



print("Loading pictures...")
# See how this works w/ Celeb Images or try your own dataset instead:
#import os
dirname = "../session-1/labdogs"
#dirname = "../../myrandompictures"
QNT=100
filenames = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]
filenames = filenames[:QNT]
assert(len(filenames) == QNT)
myimgs=np.array([plt.imread(fname)[..., :3] for fname in filenames])
myimgs = [utils.imcrop_tosquare(img_i) for img_i in myimgs]
myimgs = [resize(img_i, (100,100)) for img_i in myimgs]

imgs=np.clip(np.array(myimgs)*255, 0, 255).astype(np.uint8) # fix resize() conversion to 0..1
print("imgs.shape: ", imgs.shape)
# Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
#Xs =  tf.reshape(imgs, [1, imgs.shape[0], imgs.shape[1], imgs.shape[2]])
Xs=imgs # ya esta bolu?

print("Xs.shape: ", Xs.shape)
assert(Xs.ndim == 4 and Xs.shape[1] <= 250 and Xs.shape[2] <= 250)


ds = datasets.Dataset(Xs)
# ds = datasets.CIFAR10(flatten=False)


mean_img = ds.mean().astype(np.uint8) # bugfix adattudos
plt.title("mean")
plt.imshow(mean_img)
plt.pause(3)


std_img = ds.std().astype(np.uint8)
print("std_img.shape: ", std_img.shape)
plt.title("deviation")
plt.imshow(std_img)
plt.pause(3)


std_img = np.mean(std_img, axis=2).astype(np.uint8)
plt.title("mean of std dev across channels")
plt.imshow(std_img)
plt.pause(3)

plt.title("dataset object")
plt.imshow(ds.X[0])
plt.pause(3)

print("ds.X.shape: ", ds.X.shape)


print("batches X.shape: ")
for (X, y) in ds.train.next_batch(batch_size=10):
    print(X.shape)


# Write a function to preprocess/normalize an image, given its dataset object
# (which stores the mean and standard deviation!)
def preprocess(img, ds):
    norm_img = (img - ds.mean()) / ds.std()
    return norm_img

# Write a function to undo the normalization of an image, given its dataset object
# (which stores the mean and standard deviation!)
def deprocess(norm_img, ds):
    img = norm_img * ds.std() + ds.mean()
    return img
    
    
# Calculate the number of features in your image.
# This is the total number of pixels, or (height x width x channels).
n_features = Xs.shape[1]*Xs.shape[2]*Xs.shape[3]
print("n_features: ", n_features)


encoder_dimensions = [2048, 512, 128, 2]


X = tf.placeholder(tf.float32, [None, n_features], 'X')
assert(X.get_shape().as_list() == [None, n_features])



def encode(X, dimensions, activation=tf.nn.tanh):

    print("encode()")
    # We're going to keep every matrix we create so let's create a list to hold them all
    Ws = []

    # We'll create a for loop to create each layer:
    for layer_i, n_output in enumerate(dimensions):

        # TODO: just like in the last session,
        # we'll use a variable scope to help encapsulate our variables
        # This will simply prefix all the variables made in this scope
        # with the name we give it.  Make sure it is a unique name
        # for each layer, e.g., 'encoder/layer1', 'encoder/layer2', or
        # 'encoder/1', 'encoder/2',... 
        with tf.variable_scope("enclay_"+str(layer_i)):
        
            print("    layer: ", layer_i)

            # TODO: Create a weight matrix which will increasingly reduce
            # down the amount of information in the input by performing
            # a matrix multiplication.  You can use the utils.linear function.
            #h, W = utils.linear(x=X, n_output=n_output, activation=tf.nn.softmax) ##, name="lay_"+str(layer_i))
            #h, W = utils.linear(x=X, n_output=n_output, activation=tf.nn.relu, name="enclay_"+str(layer_i))
            h, W = utils.linear(x=X, n_output=n_output, activation=activation, name="enclay_"+str(layer_i))

            # Finally we'll store the weight matrix.
            # We need to keep track of all
            # the weight matrices we've used in our encoder
            # so that we can build the decoder using the
            # same weight matrices.
            Ws.append(W)
            
            # Replace X with the current layer's output, so we can
            # use it in the next layer.
            X = h
    
    z = X
    return Ws, z
    
    
# Then call the function
Ws, z = encode(X, encoder_dimensions)

# And just some checks to make sure you've done it right.
assert(z.get_shape().as_list() == [None, 2])
assert(len(Ws) == len(encoder_dimensions))


#[op.name for op in tf.get_default_graph().get_operations()]



#[W_i.get_shape().as_list() for W_i in Ws]
print("Ws shape:")
for W_i in Ws:
  print("    ",W_i.get_shape().as_list()) 


print("z shape: ", z.get_shape().as_list())


# We'll first reverse the order of our weight matrices
decoder_Ws = Ws[::-1]

# then reverse the order of our dimensions
# appending the last layers number of inputs.
decoder_dimensions = encoder_dimensions[::-1][1:] + [n_features]
print("decoder_dimensions: ", decoder_dimensions)

assert(decoder_dimensions[-1] == n_features)


#dja
print("Ws ", [kki.get_shape().as_list() for kki in Ws])
kk=Ws[-1::]
print("Ws[-1::] ", [kki.get_shape().as_list() for kki in kk])
kk=Ws[:-1:]
print("Ws[:-1:]", [kki.get_shape().as_list() for kki in kk])
kk=Ws[::-1]
print("Ws[::-1] ", [kki.get_shape().as_list() for kki in kk])


def decode(z, dimensions, Ws, activation=tf.nn.tanh):

    print("decode()")
    
    current_input = z
    for layer_i, n_output in enumerate(dimensions):
        # we'll use a variable scope again to help encapsulate our variables
        # This will simply prefix all the variables made in this scope
        # with the name we give it.
        with tf.variable_scope("decoder/layer/{}".format(layer_i)):

            print("    layer: ", layer_i)

            # Now we'll grab the weight matrix we created before and transpose it
            # So a 3072 x 784 matrix would become 784 x 3072
            # or a 256 x 64 matrix, would become 64 x 256
            W = tf.transpose(Ws[layer_i])

            # Now we'll multiply our input by our transposed W matrix
            h = tf.matmul(current_input, W)

            # And then use a relu activation function on its output
            current_input = activation(h)

            # We'll also replace n_input with the current n_output, so that on the
            # next iteration, our new number inputs will be correct.
            n_input = n_output
    Y = current_input
    return Y


Y = decode(z, decoder_dimensions, decoder_Ws)


##[op.name for op in tf.get_default_graph().get_operations()
## if op.name.startswith('decoder')]
#print("decoder operations:")
#for op in tf.get_default_graph().get_operations():
#	if op.name.startswith('decoder'):
#		print("    ", op.name)


print("Y shape: ", Y.get_shape().as_list())


# Calculate some measure of loss, e.g. the pixel to pixel absolute difference or squared difference
loss =tf.reduce_mean(tf.squared_difference(X, Y), 1)
print("loss shape: ", loss.get_shape().as_list() )
print("loss: ", loss)

# Now sum over every pixel and then calculate the mean over the batch dimension (just like session 2!)
# hint, use tf.reduce_mean and tf.reduce_sum
cost = tf.reduce_mean(loss)
print("cost shape: ", cost.get_shape().as_list() )


learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


print("Training...")
t1 = datetime.datetime.now()


# HASTA ACA LLEGA, LUEGO DA ERROR

# (TODO) Create a tensorflow session and initialize all of our weights:
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Some parameters for training
batch_size = QNT
n_epochs = 31
step = 1

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
examples = ds.X[:QNT]

print("preprocess...")
# We have to preprocess the images before feeding them to the network.
# I'll do this once here, so we don't have to do it every iteration.
test_examples = preprocess(examples, ds).reshape(-1, n_features)

# If we want to just visualize them, we can create a montage.
test_images = utils.montage(examples).astype(np.uint8)

# Store images so we can make a gif
gifs = []

# Now for our training:
for epoch_i in range(n_epochs):

    print("epoch: ", epoch_i)
    
    # Keep track of the cost
    this_cost = 0
    
    # Iterate over the entire dataset in batches
    for batch_X, _ in ds.train.next_batch(batch_size=batch_size):
        
        # (TODO) Preprocess and reshape our current batch, batch_X:
        this_batch = preprocess(batch_X, ds).reshape(-1, n_features)
        
        # Compute the cost, and run the optimizer.
        this_cost += sess.run([cost, optimizer], feed_dict={X: this_batch})[0]
        
        print("    batch cost: ", this_cost)
    
    # Average cost of this epoch
    avg_cost = this_cost / ds.X.shape[0] / batch_size
    print("epoch/avg_cost: ", epoch_i, avg_cost)
    
    # Let's also try to see how the network currently reconstructs the input.
    # We'll draw the reconstruction every `step` iterations.
    if epoch_i % step == 0:
        
        # (TODO) Ask for the output of the network, Y, and give it our test examples
        recon = sess.run(Y, feed_dict={X: test_examples})
                         
        # Resize the 2d to the 4d representation:
        rsz = recon.reshape(examples.shape)

        # We have to unprocess the image now, removing the normalization
        unnorm_img = deprocess(rsz, ds)
                         
        # Clip to avoid saturation
        clipped = np.clip(unnorm_img, 0, 255)

        # And we can create a montage of the reconstruction
        recon = utils.montage(clipped).astype(np.uint8)
        
        # Store for gif
        gifs.append(recon)

        #fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        #axs[0].imshow(test_images)
        #axs[0].set_title('Original')
        #axs[1].imshow(recon)
        #axs[1].set_title('Synthesis')
        #fig.canvas.draw()
        plt.title("recon "+str(epoch_i))
        plt.imshow(recon)
        plt.show()
        plt.pause(1)


t2 = datetime.datetime.now()
delta = t2 - t1
print("             Total training time: ", delta.total_seconds())

_ = gif.build_gif(gifs, saveto='gif_training_'+TID+'.gif', interval=0.3, show_gif=False)


#fig, axs = plt.subplots(1, 2, figsize=(10, 10))
#axs[0].imshow(test_images)
#axs[0].set_title('Original')
#axs[1].imshow(recon)
#axs[1].set_title('Synthesis')
#fig.canvas.draw()
plt.title("Synthesis")
plt.imshow(recon)
plt.show()
plt.pause(5)
plt.close()

plt.imsave(arr=test_images, fname='test_s3b01_'+TID+'.png')
plt.imsave(arr=recon, fname='recon_s3b01_'+TID+'.png')

#
# Visualize the Embedding
#

zs = sess.run(z, feed_dict={X:test_examples})


print("zs.shape: ", zs.shape)


plt.title("scatter")
plt.scatter(zs[:, 0], zs[:, 1])
plt.show()
plt.pause(3)


n_images = QNT
idxs = np.linspace(np.min(zs) * 2.0, np.max(zs) * 2.0,
                   int(np.ceil(np.sqrt(n_images))))
xs, ys = np.meshgrid(idxs, idxs)
grid = np.dstack((ys, xs)).reshape(-1, 2)[:n_images,:]


fig, axs = plt.subplots(1,2,figsize=(8,3))
axs[0].scatter(zs[:, 0], zs[:, 1],
               edgecolors='none', marker='o', s=2)
axs[0].set_title('Autoencoder Embedding')
axs[1].scatter(grid[:,0], grid[:,1],
               edgecolors='none', marker='o', s=2)
axs[1].set_title('Ideal Grid')

plt.show()
plt.pause(3)

from scipy.spatial.distance import cdist
cost = cdist(grid[:, :], zs[:, :], 'sqeuclidean')
from scipy.optimize._hungarian import linear_sum_assignment
indexes = linear_sum_assignment(cost)


print("indexes: ", indexes)


plt.title("indexes")
plt.figure(figsize=(3, 3))
for i in range(len(zs)):
    plt.plot([zs[indexes[1][i], 0], grid[i, 0]],
             [zs[indexes[1][i], 1], grid[i, 1]], 'r')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()
plt.pause(3)

examples_sorted = []
for i in indexes[1]:
    examples_sorted.append(examples[i])
plt.figure(figsize=(5, 5))
img = utils.montage(np.array(examples_sorted)).astype(np.uint8)
plt.title("sorted imgs")
plt.imshow(img,
           interpolation='nearest')
plt.imsave(arr=img, fname='sorted_s3b01_'+TID+'.png')
plt.pause(3)


#
#2D Latent Manifold
#

# This is a quick way to do what we could have done as
# a nested for loop:
SQ=int(np.sqrt(QNT))
zs = np.meshgrid(np.linspace(-1, 1, SQ),
                 np.linspace(-1, 1, SQ))

# Now we have 100 x 2 values of every possible position
# in a 2D grid from -1 to 1:
zs = np.c_[zs[0].ravel(), zs[1].ravel()]

#print("zs: ", zs)
#print("")

recon = sess.run(Y, feed_dict={z:zs})
#recon = decode(zs, decoder_dimensions, decoder_Ws)

# reshape the result to an image:
rsz = recon.reshape(examples.shape)

# Deprocess the result, unnormalizing it
unnorm_img = deprocess(rsz, ds)

# clip to avoid saturation
clipped = np.clip(unnorm_img, 0, 255)

# Create a montage
img_i = utils.montage(clipped).astype(np.uint8)


plt.figure(figsize=(5, 5))
plt.title("manifold")
plt.imshow(img_i)
plt.imsave(arr=img_i, fname='manifold_s3b01_'+TID+'.png')
plt.pause(3)



#eop


