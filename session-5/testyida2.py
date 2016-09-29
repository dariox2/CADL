

print("Loading tensorflow...")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


plt.style.use('bmh')
plt.ion()
plt.figure(figsize=(3, 3))



def create_input_pipeline_yida(files, batch_size, n_epochs, shape, crop_shape=None,
                          crop_factor=1.0, n_threads=2, seed=None):
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

    

    # Now we'll create a batch generator that will also shuffle our examples.
    # We tell it how many it should have in its buffer when it randomly
    # permutes the order.
    min_after_dequeue = len(files) // 5

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (n_threads + 1) * batch_size

    # Randomize the order and output batches of batch_size.
    batch = tf.train.shuffle_batch([crops],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=n_threads,
                                   seed=seed) #, shapes=(64,64,3))

    # alternatively, we could use shuffle_batch_join to use multiple reader
    # instances, or set shuffle_batch's n_threads to higher than 1.

    return batch




def CELEByida(path='./img_align_celeba/'):
  fs = []
  for f in os.listdir(path):
    if f.endswith('.jpg'):
      fs=fs+f
  return fs



print("Loading celebrities...")
from libs.datasets import CELEB
files = CELEByida("../session-1/img_align_celeba/") # only 100


from libs.dataset_utils import create_input_pipeline
batch_size = 5
n_epochs = 3
input_shape = [218, 178, 3]
crop_shape = [64, 64, 3]
crop_factor = 0.8

seed=15

batch = create_input_pipeline_yida(
    files=files,
    batch_size=batch_size,
    n_epochs=n_epochs,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape,
    seed=seed)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


batch_xs = sess.run(batch)
print("batch_xs shape: ", batch_xs.shape)
print("batch_xs dtype: ", batch_xs.dtype, "  max: ", np.max(batch_xs.dtype))
for i in range(0,3):
  img= (batch_xs[i] / np.max(np.abs(batch_xs[i])) * 128 + 128).astype(np.uint8)
  plt.imshow(img)
  plt.pause(1)

# eop

