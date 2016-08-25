
#
# test list
#

import numpy as np
import tensorflow as tf

np.set_printoptions(threshold=np.inf) # display FULL array (infinite)

a=[1,2,3]
b=[4,5,6]

p=[a, b]
print(p)

q=np.array(p)
print(q)

r=np.resize(q, None)


