
#
# test shuffle_batch
#
# version with 2 seeds, 1 for graph, 1 for shuffle
# and with 2 separate pipelines
#
# result: only the 1st batch matches (?)
#

print("Loading tensorflow...")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


from libs import utils
import datetime


tf.set_random_seed(1)


def create_input_pipeline_yida(files, batch_size, n_epochs, shape, crop_shape=None,
                          crop_factor=1.0, n_threads=1, seed=None):
    """Creates a pipefile from a list of image files.
    Includes batch generator/central crop/resizing options.
    The resulting generator will dequeue the images batch_size at a time until
    it throws tf.errors.OutOfRangeError when there are no more images left in
    the queue.

    Parameters
    ----------
    files : list
        List of paths to image files.
    batch_size : int
        Number of image files to load at a time.
    n_epochs : int
        Number of epochs to run before raising tf.errors.OutOfRangeError
    shape : list
        [height, width, channels]
    crop_shape : list
        [height, width] to crop image to.
    crop_factor : float
        Percentage of image to take starting from center.
    n_threads : int, optional
        Number of threads to use for batch shuffling
    """

    # We first create a "producer" queue.  It creates a production line which
    # will queue up the file names and allow another queue to deque the file
    # names all using a tf queue runner.
    # Put simply, this is the entry point of the computational graph.
    # It will generate the list of file names.
    # We also specify it's capacity beforehand.
    producer = tf.train.string_input_producer(
        files, capacity=len(files))

    # We need something which can open the files and read its contents.
    reader = tf.WholeFileReader()

    # We pass the filenames to this object which can read the file's contents.
    # This will create another queue running which dequeues the previous queue.
    keys, vals = reader.read(producer)

    # And then have to decode its contents as we know it is a jpeg image
    imgs = tf.image.decode_jpeg(
        vals,
        channels=3)

    # We have to explicitly define the shape of the tensor.
    # This is because the decode_jpeg operation is still a node in the graph
    # and doesn't yet know the shape of the image.  Future operations however
    # need explicit knowledge of the image's shape in order to be created.
    imgs.set_shape(shape)

    # Next we'll centrally crop the image to the size of 100x100.
    # This operation required explicit knowledge of the image's shape.
    if shape[0] > shape[1]:
        rsz_shape = [int(shape[0] / shape[1] * crop_shape[0] / crop_factor),
                     int(crop_shape[1] / crop_factor)]
    else:
        rsz_shape = [int(crop_shape[0] / crop_factor),
                     int(shape[1] / shape[0] * crop_shape[1] / crop_factor)]
    rszs = tf.image.resize_images(imgs, rsz_shape[0], rsz_shape[1])
    crops = (tf.image.resize_image_with_crop_or_pad(
        rszs, crop_shape[0], crop_shape[1])
        if crop_shape is not None
        else imgs)

    print("crops.shape: ", crops)

    # Now we'll create a batch generator that will also shuffle our examples.
    # We tell it how many it should have in its buffer when it randomly
    # permutes the order.
    min_after_dequeue = len(files) // 5

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (n_threads + 1) * batch_size

    print("crops.get_shape(): ",crops.get_shape())
    # Randomize the order and output batches of batch_size.
    batch = tf.train.shuffle_batch([crops],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=n_threads,
                                   seed=seed, shapes=(64,64,3))

    # alternatively, we could use shuffle_batch_join to use multiple reader
    # instances, or set shuffle_batch's n_threads to higher than 1.

    return batch




def CELEByida(path='./img_align_celeba/'):
  fs = [os.path.join(path, f)
  for f in os.listdir(path) if f.endswith('.jpg')]
  fs=sorted(fs)
  return fs



print("Loading celebrities...")
from libs.datasets import CELEB
files = CELEByida("../session-1/img_align_celeba/") # only 100

#print("files=",files)


from libs.dataset_utils import create_input_pipeline
batch_size = 8
n_epochs = 3
input_shape = [218, 178, 3]
crop_shape = [64, 64, 3]
crop_factor = 0.8

seed=15

files2=files.copy()

tf.set_random_seed(1)

batch1 = create_input_pipeline_yida(
    files=files,
    batch_size=batch_size,
    n_epochs=n_epochs,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape,
    seed=seed)

tf.set_random_seed(1)

batch2 = create_input_pipeline_yida(
    files=files2,
    batch_size=batch_size,
    n_epochs=n_epochs,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape,
    seed=seed)


mntg=[]

sess1 = tf.Session()

coord1 = tf.train.Coordinator()
threads1 = tf.train.start_queue_runners(sess=sess1, coord=coord1)

batch_xs1 = sess1.run(batch1)

for i in range(0,len(batch_xs1)):
  img= batch_xs1[i] / 255.0
  mntg.append(img)

batch_xs2 = sess1.run(batch2)

for i in range(0,len(batch_xs2)):
  img= batch_xs2[i] / 255.0
  mntg.append(img)


TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")
m=utils.montage(mntg, saveto="montage_"+TID+".png")

plt.figure(figsize=(5, 5))
plt.imshow(m)
plt.show()

# eop

