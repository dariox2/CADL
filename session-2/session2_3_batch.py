
#
# Training a Network w/Tensorflow
#
# PART 3 - Multiple images
#
# refer:
#    4 imgs, 128x128, 10 iter: biswal home 38.9s;   cyclop (vm):  7.7s
#   16 imgs, 128x128,  3 iter: biswal home 46.8s;   cyclop (vm):  8.3s
#   16 imgs, 128x128, 10 iter: biswal home: 151.8s; cyclop (vm): 30.2s
#   16 imgs, 128x128, 1k iter: cyclop (centos vm): 3058s (~51m)
#                                                  2098s (bugfix gif too large)
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import tensorflow as tf
from libs import gif, utils
import IPython.display as ipyd
from datetime import datetime

#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)

plt.ion()


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


def build_model(xs, ys, n_neurons, n_layers, activation_fn,
                final_activation_fn, cost_type):
    
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    if xs.ndim != 2:
        raise ValueError(
            'xs should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
    if ys.ndim != 2:
        raise ValueError(
            'ys should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
        
    n_xs = xs.shape[1]
    n_ys = ys.shape[1]
    
    X = tf.placeholder(name='X', shape=[None, n_xs],
                       dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, n_ys],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = utils.linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='layer{}'.format(layer_i))[0]

    Y_pred = utils.linear(
        current_input, n_ys,
        activation=final_activation_fn,
        name='pred')[0]
    
    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.abs(Y - Y_pred), 1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(Y, Y_pred), 1))
    else:
        raise ValueError(
            'Unknown cost_type: {}.  '.format(
            cost_type) + 'Use only "l1_norm" or "l2_norm"')
    
    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}


def train(imgs,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=10,
          gif_step=2,
          n_neurons=30,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):

    N, H, W, C = imgs.shape
    all_xs, all_ys = [], []
    for img_i, img in enumerate(imgs):
        xs, ys = split_image(img)
        all_xs.append(np.c_[xs, np.repeat(img_i, [xs.shape[0]])])
        all_ys.append(ys)
    xs = np.array(all_xs).reshape(-1, 3)
    xs = (xs - np.mean(xs, 0)) / np.std(xs, 0)
    ys = np.array(all_ys).reshape(-1, 3)
    ys = ys / 127.5 - 1

    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn,
                            cost_type)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model['cost'])
        sess.run(tf.initialize_all_variables())
        gifs = []
        costs = []
        step_i = 0
        for it_i in range(n_iterations):
            # Get a random sampling of the dataset
            idxs = np.random.permutation(range(len(xs)))

            # The number of batches we have to iterate over
            n_batches = len(idxs) // batch_size
            training_cost = 0

            # Now iterate over our stochastic minibatches:
            for batch_i in range(n_batches):

                # Get just minibatch amount of data
                idxs_i = idxs[batch_i * batch_size:
                              (batch_i + 1) * batch_size]

                # And optimize, also returning the cost so we can monitor
                # how our optimization is doing.
                cost = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: xs[idxs_i],
                               model['Y']: ys[idxs_i]})[0]
                training_cost += cost

            print('iteration {}/{}: cost {}'.format(
                    it_i + 1, n_iterations, training_cost / n_batches))

            # Also, every 20 iterations, we'll draw the prediction of our
            # input xs, which should try to recreate our image!
            if (it_i + 1) % gif_step == 0:
                costs.append(training_cost / n_batches)
                ys_pred = model['Y_pred'].eval(
                    feed_dict={model['X']: xs}, session=sess)
                img = ys_pred.reshape(imgs.shape)
                gifs.append(img)
        return gifs

import urllib
def get_celeb_files(qfil):
    if not os.path.exists('img_align_celeba'):
        os.mkdir('img_align_celeba')

    for img_i in range(1, qfil+1):

        f = '000%03d.jpg' % img_i

        url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f

        print(url, end='\r')

        urllib.request.urlretrieve(url, os.path.join('img_align_celeba', f))

    files = [os.path.join('img_align_celeba', file_i)
             for file_i in os.listdir('img_align_celeba')
             if '.jpg' in file_i]
    return files[0:qfil]


def get_celeb_imgs(qpic):
    """
    Returns
    -------
    imgs : list of np.ndarray
        List of the first <qpic> images from the celeb dataset
    """
    return [plt.imread(f_i) for f_i in get_celeb_files(qpic)]


#############################################################
#
# MAIN
#

print("Reading images...")
QNT=16
switchcelebs=False
if switchcelebs:
  celeb_imgs = np.array(get_celeb_imgs(QNT))
  plt.figure(figsize=(6, 6))
  print(celeb_imgs)
  print (np.array(celeb_imgs).shape)
  pltdataset=utils.montage(celeb_imgs, saveto="batch2_3_temp_dataset.png").astype(np.uint8)
  plt.imshow(pltdataset)
  plt.imsave(fname='batch2_3_dataset.png', arr=pltdataset)
  trainimgs = np.array(celeb_imgs).copy()
else:
  dirname = "labdogs"
  filenames = [os.path.join(dirname, fname)
            for fname in os.listdir(dirname)]
  filenames = filenames[:QNT]
  assert(len(filenames) == QNT)
  #myimgs = [plt.imread(fname)[..., :3] for fname in filenames]
  myimgs=np.array([plt.imread(fname) for fname in filenames])
  myimgs = [utils.imcrop_tosquare(img_i) for img_i in myimgs]
  myimgs = [resize(img_i, (128,128)) for img_i in myimgs]
  myimgs=np.clip(np.array(myimgs)*255, 0, 255).astype(np.uint8) # fix resize() conversion to 0..1
  pltdataset=utils.montage(myimgs, saveto="batch2_3_temp_dataset.png").astype(np.uint8)
  plt.imshow(pltdataset)
  plt.imsave(fname='batch2_3_dataset.png', arr=pltdataset)
  trainimgs = np.array(myimgs).copy()


plt.show()
plt.pause(1)

print("Training...")
t1 = datetime.now()
trainedgifs = train(imgs=trainimgs, n_iterations=1000, gif_step=50)
t2 = datetime.now()
delta = t2 - t1
print("             Total training time: ", delta.total_seconds())
plt.close()
print("Saving results...")
montage_gifs = [np.clip(utils.montage(
                (m * 127.5) + 127.5, saveto='batch2_3_montage_temp.png'), 0, 255).astype(np.uint8)
                    for m in trainedgifs]
_ = gif.build_gif(montage_gifs, saveto='batch2_3_multiple.gif')

plt.show()
plt.pause(5)
plt.close()

final = trainedgifs[-1]
final_gif = [np.clip(((m * 127.5) + 127.5), 0, 255).astype(np.uint8) for m in final]
gif.build_gif(final_gif, saveto='batch2_3_final.gif')

#plt.imshow(_)
plt.show()
plt.pause(5)
plt.close()


# eop


