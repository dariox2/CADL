
#
# test03
#
# image resize bug test
#
# http://stackoverflow.com/questions/37032251/tensorflow-image-resize-mess-up-image-on-unknown-image-size
#

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

file_contents = tf.read_file('fot1.jpg')
im = tf.image.decode_jpeg(file_contents)
im = tf.image.resize_images(im, 256, 256)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

img = sess.run(im)

plt.imshow(img.astype(np.uint8))
plt.show()

# eop


