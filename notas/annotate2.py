#
# example (2): annotate figure with text
#

#import numpy as np
import matplotlib.pyplot as plt

#fig = plt.figure()

img = plt.imread('../test/babyfox1.jpg')
#fig.annotate('local max', xy=(2, 1), xytext=(3, 1.5),             arrowprops=dict(facecolor='black', shrink=0.05), )

for i in range(10, 20):
  for j in range(10, 100):
    img[i,j]=[128,128,128]

plt.imshow(img)
plt.show()

plt.imsave(fname='annotate2_test.png', arr=img)


