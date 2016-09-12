#
# example (2): annotate figure with text
#

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

img = plt.imread('../test/babyfox1.jpg')
#fig.annotate('local max', xy=(2, 1), xytext=(3, 1.5),             arrowprops=dict(facecolor='black', shrink=0.05), )

plt.text(0.1, 0.9, 'matplotlib', ha='center', va='center')   #transform=plt.transAxes)

plt.imsave(fname='annotate2_test.png', arr=img)


#ax = fig.add_subplot(111)

#t = np.arange(0.0, 5.0, 0.01)
#s = np.cos(2*np.pi*t)
#line, = ax.plot(t, s, lw=2)

#ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),             arrowprops=dict(facecolor='black', shrink=0.05),             )

#ax.set_ylim(-2,2)


#plt.text(0.1, 0.9, 'matplotlib', ha='center', va='center', transform=ax.transAxes)

#fig.savefig('annotate2_text.png', bbox_inches='tight')

plt.show()

