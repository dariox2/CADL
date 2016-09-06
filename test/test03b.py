
#
# test03 (b)
#
# image resize bug test
#
# http://stackoverflow.com/questions/37032251/tensorflow-image-resize-mess-up-image-on-unknown-image-size
#

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

file_contents = tf.read_file('fot1.jpg')
im = tf.image.decode_jpeg(file_contents)
print(im)
im = tf.image.resize_images(im, 256, 256)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

img = sess.run(im)

img=(img.astype(np.uint8)) # discard decimal part...
img=img / 255.0 # ...and normalize 0-255 to floats in the 0.0-1.0 range
print(img)
plt.imshow(img)
plt.show()

input("press any key...")



# eop


