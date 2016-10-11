
#
# Session 5, part 3
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

def wait(n):
    #plt.pause(n)
    plt.pause(3)
    input("(press enter)")


##
## Part 3 - Latent-Space Arithmetic
##


#
# Loading the Pre-Trained Model
#

# We're now going to work with a pre-trained VAEGAN model on the
# Celeb Net dataset. Let's load this model:


tf.reset_default_graph()

print("Import vaegan model...")
from libs import celeb_vaegan as CV
net = CV.get_celeb_vaegan_model()


# We'll load the graph_def contained inside this dictionary. It
# follows the same idea as the `inception`, `vgg16`, and `i2v`
# pretrained networks. It is a dictionary with the key `graph_def`
# defined, with the graph's pretrained network. It also includes
# `labels` and a `preprocess` key. We'll have to do one additional
# thing which is to turn off the random sampling from variational
# layer. This isn't really necessary but will ensure we get the same
# results each time we use the network. We'll use the `input_map`
# argument to do this. Don't worry if this doesn't make any sense, as
# we didn't cover the variational layer in any depth. Just know that
# this is removing a random process from the network so that it is
# completely deterministic. If we hadn't done this, we'd get slightly
# different results each time we used the network (which may even be
# desirable for your purposes).

sess = tf.Session()
g = tf.get_default_graph()
print("import graph_def...")
tf.import_graph_def(net['graph_def'], name='net', input_map={
        'encoder/variational/random_normal:0': np.zeros(512, dtype=np.float32)})

#for op in g.get_operations():
#  print(op.name)


# Now let's get the relevant parts of the network: `X`, the input
# image to the network, `Z`, the input image's encoding, and `G`, the
# decoded image. In many ways, this is just like the Autoencoders we
# learned about in Session 3, except instead of `Y` being the output,
# we have `G` from our generator! And the way we train it is very
# different: we use an adversarial process between the generator and
# discriminator, and use the discriminator's own distance measure to
# help train the network, rather than pixel-to-pixel differences.

X = g.get_tensor_by_name('net/x:0')
Z = g.get_tensor_by_name('net/encoder/variational/z:0')
G = g.get_tensor_by_name('net/generator/x_tilde:0')


# Let's get some data to play with:

files = datasets.CELEB()
#img_i = 50
#img = plt.imread(files[img_i])
#plt.imshow(img)
#plt.title("some celeb")
#wait(1)

# Now preprocess the image, and see what the generated image looks
# like (i.e. the lossy version of the image through the network's
# encoding and decoding).

#p = CV.preprocess(img)
#synth = sess.run(G, feed_dict={X: p[np.newaxis]})

#fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#axs[0].imshow(p)
#plt.imshow(synth[0] / synth.max())
#plt.title("lossy version")
#wait(1)


# So we lost a lot of details but it seems to be able to express
# quite a bit about the image. Our inner most layer, `Z`, is only 512
# values yet our dataset was 200k images of 64 x 64 x 3 pixels (about
# 2.3 GB of information). That means we're able to express our nearly
# 2.3 GB of information with only 512 values! Having some loss of
# detail is certainly expected!
#
# <a name="exploring-the-celeb-net-attributes"></a>
# ## Exploring the Celeb Net Attributes
#
# Let's now try and explore the attributes of our dataset. We didn't
# train the network with any supervised labels, but the Celeb Net
# dataset has 40 attributes for each of its 200k images. These are
# already parsed and stored for you in the `net` dictionary:


print("net keys: ", net.keys())


len(net['labels'])


print("net labels: ", net['labels'])


# Let's see what attributes exist for one of the celeb images:

#plt.title("attributes")
#plt.imshow(img)
#print("attributes of ", img_i)
#[net['labels'][i] for i, attr_i in enumerate(net['attributes'][img_i]) if attr_i]
#for i, attr_i in enumerate(net['attributes'][img_i]):
#  if attr_i:
#    print(i, net['labels'][i])
#wait(1)


#
# Find the Latent Encoding for an Attribute
#

# The Celeb Dataset includes attributes for each of its 200k+ images.
# This allows us to feed into the encoder some images that we know
# have a *specific* attribute, e.g. "smiling". We store what their
# encoding is and retain this distribution of encoded values. We can
# then look at any other image and see how it is encoded, and
# slightly change the encoding by adding the encoded of our smiling
# images to it! The result should be our image but with more smiling.
# That is just insane and we're going to see how to do it. First lets
# inspect our latent space:

print("Z shape: ", Z.get_shape())


# We have 512 features that we can encode any image with. Assuming
# our network is doing an okay job, let's try to find the `Z` of the
# first 100 images with the 'Bald' attribute:

bald_label = net['labels'].index('Big_Nose')

print("bald_label: ", bald_label)


# Let's get all the bald image indexes:

bald_img_idxs = np.where(net['attributes'][:, bald_label])[0]


print("bald img idxs: ", bald_img_idxs)
print("bald idxs len: ", len(bald_img_idxs))

# Now let's just load 100 of their images:

print("big nose #100: ", bald_img_idxs[99])

bald_imgs = [plt.imread(files[bald_img_i])[..., :3]
             for bald_img_i in bald_img_idxs[:100]]

print("bald imgs len: ", len(bald_imgs))

# Let's see if the mean image looks like a good bald person or not:

plt.title("bald person")
plt.imshow(np.mean(bald_imgs, 0).astype(np.uint8))
wait(1)

# Yes that is definitely a bald person. Now we're going to try to
# find the encoding of a bald person. One method is to try and find
# every other possible image and subtract the "bald" person's latent
# encoding. Then we could add this encoding back to any new image and
# hopefully it makes the image look more bald. Or we can find a bunch
# of bald people's encodings and then average their encodings
# together. This should reduce the noise from having many different
# attributes, but keep the signal pertaining to the baldness.
#
# Let's first preprocess the images:

bald_p = np.array([CV.preprocess(bald_img_i) for bald_img_i in bald_imgs])


# Now we can find the latent encoding of the images by calculating
# `Z` and feeding `X` with our `bald_p` images:
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


bald_zs = sess.run(Z, feed_dict={X: bald_p}) # dja


# Now let's calculate the mean encoding:


bald_feature = np.mean(bald_zs, 0, keepdims=True)

print("bald feature shape: ", bald_feature.shape)


# Let's try and synthesize from the mean bald feature now and see how
# it looks:
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


bald_generated = sess.run(G, feed_dict={Z: bald_feature}) # dja


plt.title("bald generated")
plt.imshow(bald_generated[0] / bald_generated.max())
wait(1)


#
# Latent Feature Arithmetic
#

# Let's now try to write a general function for performing everything
# we've just done so that we can do this with many different
# features. We'll then try to combine them and synthesize people with
# the features we want them to have...

def get_features_for(label='Bald', has_label=True, n_imgs=50):
    label_i = net['labels'].index(label)
    label_idxs = np.where(net['attributes'][:, label_i] == has_label)[0]
    label_idxs = np.random.permutation(label_idxs)[:n_imgs]
    imgs = [plt.imread(files[img_i])[..., :3]
            for img_i in label_idxs]
    preprocessed = np.array([CV.preprocess(img_i) for img_i in imgs])
    zs = sess.run(Z, feed_dict={X: preprocessed})
    return np.mean(zs, 0)


# Let's try getting some attributes positive and negative features.
# Be sure to explore different attributes! Also try different values
# of `n_imgs`, e.g. 2, 3, 5, 10, 50, 100. What happens with different
# values?
#
# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


# Explore different attributes
z1 = get_features_for('Attractive', True, n_imgs=10)
z2 = get_features_for('Attractive', False, n_imgs=10)
z3 = get_features_for('Chubby', True, n_imgs=10)
z4 = get_features_for('Chubby', False, n_imgs=10)


b1 = sess.run(G, feed_dict={Z: z1[np.newaxis]})
b2 = sess.run(G, feed_dict={Z: z2[np.newaxis]})
b3 = sess.run(G, feed_dict={Z: z3[np.newaxis]})
b4 = sess.run(G, feed_dict={Z: z4[np.newaxis]})


plt.close()
fig, axs = plt.subplots(1, 5, figsize=(9, 4))
plt.suptitle("attract / not attract / chubby / not chubby")
axs[0].imshow(b1[0] / b1.max()),  axs[0].grid('off'), axs[0].axis('off')
axs[1].imshow(b2[0] / b2.max()),  axs[1].grid('off'), axs[1].axis('off')
axs[2].imshow(b3[0] / b3.max()), axs[2].grid('off'), axs[2].axis('off')
axs[3].imshow(b4[0] / b4.max()),  axs[3].grid('off'), axs[3].axis('off')

wait(1)
plt.cla()


# Now let's interpolate between the "Male" and "Not Male" categories:

notmale_vector = z2 - z1
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z1 + notmale_vector*amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})

plt.suptitle("attract ... not attract")
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')

wait(1)
plt.cla()

# And the same for smiling:

smiling_vector = z3 - z4
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z4 + smiling_vector*amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})

plt.suptitle("not chubby ... chubby")
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i] / g[i].max(), 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')

wait(1)
plt.cla()

# There's also no reason why we have to be within the boundaries of
# 0-1. We can extrapolate beyond, in, and around the space.

plt.suptitle("extrapolate")
n_imgs = 5
amt = np.linspace(-1.5, 2.5, n_imgs)
zs = np.array([z4 + smiling_vector*amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    #ax_i.grid('off')
    ax_i.axis('off')


wait(1)
plt.cla()

#
# Extensions
#

# [Tom White](https://twitter.com/dribnet), Lecturer at Victoria
# University School of Design, also recently demonstrated an
# alternative way of interpolating using a sinusoidal interpolation.
# He's created some of the most impressive generative images out
# there and luckily for us he has detailed his process in the arxiv
# preprint: https://arxiv.org/abs/1609.04468 - as well, be sure to
# check out his twitter bot, https://twitter.com/smilevector - which
# adds smiles to people :) - Note that the network we're using is
# only trained on aligned faces that are frontally facing, though
# this twitter bot is capable of adding smiles to any face. I suspect
# that he is running a face detection algorithm such as AAM, CLM, or
# ASM, cropping the face, aligning it, and then running a similar
# algorithm to what we've done above. Or else, perhaps he has trained
# a new model on faces that are not aligned. In any case, it is well
# worth checking out!
#
# Let's now try and use sinusoidal interpolation using his
# implementation in
# [plat](https://github.com/dribnet/plat/blob/master/plat/interpolate.py#L16-L24)
# which I've copied below:


def slerp(val, low, high):
    # Spherical interpolation. val has a range of 0 to 1.
    if val <= 0:
        return low
    elif val >= 1:
        return high
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


plt.suptitle("sinusoidal interp")
amt = np.linspace(0, 1, n_imgs)
zs = np.array([slerp(amt_i, z1, z2) for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')

wait(1)
plt.cla()

# It's certainly worth trying especially if you are looking to
# explore your own model's latent space in new and interesting ways.
#
# Let's try and load an image that we want to play with. We need an
# image as similar to the Celeb Dataset as possible. Unfortunately,
# we don't have access to the algorithm they used to "align" the
# faces, so we'll need to try and get as close as possible to an
# aligned face image. One way you can do this is to load up one of
# the celeb images and try and align an image to it using e.g.
# Photoshop or another photo editing software that lets you blend and
# move the images around. That's what I did for my own face...


img = plt.imread('parag.png')[..., :3]
img = CV.preprocess(img, crop_factor=1.0)[np.newaxis]

# Let's see how the network encodes it:

plt.suptitle("blurry Parag")
img_ = sess.run(G, feed_dict={X: img})
#fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plt.cla()
for i, ax_i in enumerate(axs):
  ax_i.cla()
  ax_i.grid('off')
  ax_i.axis('off')
  
axs[0].imshow(img[0])
axs[1].imshow(np.clip(img_[0] / np.max(img_), 0, 1))

wait(1)
plt.cla()

# Notice how blurry the image is. Tom White's preprint suggests one
# way to sharpen the image is to find the "Blurry" attribute vector:

z1 = get_features_for('Blurry', True, n_imgs=25)
z2 = get_features_for('Blurry', False, n_imgs=25)
unblur_vector = z2 - z1


z = sess.run(Z, feed_dict={X: img})

plt.suptitle("unblur vector")
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z[0] + unblur_vector * amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i] / g[i].max(), 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')

wait(1)
plt.cla()

# Notice that the image also gets brighter and perhaps other features
# than simply the bluriness of the image changes. Tom's preprint
# suggests that this is due to the correlation that blurred images
# have with other things such as the brightness of the image,
# possibly due biases in labeling or how photographs are taken. He
# suggests that another way to unblur would be to synthetically blur
# a set of images and find the difference in the encoding between the
# real and blurred images. We can try it like so:


from scipy.ndimage import gaussian_filter

idxs = np.random.permutation(range(len(files)))
imgs = [plt.imread(files[idx_i]) for idx_i in idxs[:100]]
blurred = []
for img_i in imgs:
    img_copy = np.zeros_like(img_i)
    for ch_i in range(3):
        img_copy[..., ch_i] = gaussian_filter(img_i[..., ch_i], sigma=3.0)
    blurred.append(img_copy)


# Now let's preprocess the original images and the blurred ones
imgs_p = np.array([CV.preprocess(img_i) for img_i in imgs])
blur_p = np.array([CV.preprocess(img_i) for img_i in blurred])

# And then compute each of their latent features
noblur = sess.run(Z, feed_dict={X: imgs_p})
blur = sess.run(Z, feed_dict={X: blur_p})


synthetic_unblur_vector = np.mean(noblur - blur, 0)


plt.suptitle("synthetic unblur vector")
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z[0] + synthetic_unblur_vector * amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')

wait(1)
plt.cla()


# For some reason, it also doesn't like my glasses very much. Let's
# try and add them back.


z1 = get_features_for('Eyeglasses', True)
z2 = get_features_for('Eyeglasses', False)
glass_vector = z1 - z2


z = sess.run(Z, feed_dict={X: img})


plt.suptitle("glass vector")
n_imgs = 5
amt = np.linspace(0, 1, n_imgs)
zs = np.array([z[0] + glass_vector * amt_i + unblur_vector * amt_i for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')


wait(1)
plt.cla()


# Well, more like sunglasses then. Let's try adding everything in
# there now!


plt.suptitle("everything")
n_imgs = 5
amt = np.linspace(0, 1.0, n_imgs)
zs = np.array([z[0] + glass_vector * amt_i + unblur_vector * amt_i + amt_i * smiling_vector for amt_i in amt])
g = sess.run(G, feed_dict={Z: zs})
#fig, axs = plt.subplots(1, n_imgs, figsize=(20, 4))
for i, ax_i in enumerate(axs):
    ax_i.imshow(np.clip(g[i], 0, 1))
    ax_i.grid('off')
    ax_i.axis('off')


wait(1)
plt.cla()

# Well it was worth a try anyway. We can also try with a lot of
# images and create a gif montage of the result:


print("creating montage...")
n_imgs = 5
amt = np.linspace(0, 1.5, n_imgs)
z = sess.run(Z, feed_dict={X: imgs_p})
imgs = []
for amt_i in amt:
    zs = z + synthetic_unblur_vector * amt_i + amt_i * smiling_vector
    g = sess.run(G, feed_dict={Z: zs})
    m = utils.montage(np.clip(g, 0, 1))
    imgs.append(m)


gif.build_gif(imgs, saveto='celeb_unblur_chubby.gif', interval=0.2)


#ipyd.Image(url='celeb.gif?i={}'.format(np.random.rand()), height=1000, width=1000)


# Exploring multiple feature vectors and applying them to images from
# the celeb dataset to produce animations of a face, saving it as a
# GIF. Recall you can store each image frame in a list and then use
# the `gif.build_gif` function to create a gif. Explore your own
# syntheses and then include a gif of the different images you create
# as "celeb.gif" in the final submission. Perhaps try finding
# unexpected synthetic latent attributes in the same way that we
# created a blur attribute. You can check the documentation in
# scipy.ndimage for some other image processing techniques, for
# instance: http://www.scipy-lectures.org/advanced/image_processing/
# - and see if you can find the encoding of another attribute that
# you then apply to your own images. You can even try it with many
# images and use the `utils.montage` function to create a large grid
# of images that evolves over your attributes. Or create a set of
# expressions perhaps. Up to you just explore!
#

# <h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>
#... DO SOMETHING AWESOME ! ... #
#dja
#imgs = []
#gif.build_gif(imgs=imgs, saveto='vaegan.gif')
wait(1)


# Please visit [session-5-part2.ipynb](session-5-part2.ipynb) for the
# rest of the homework!

# eop
