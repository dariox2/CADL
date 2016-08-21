
import numpy as np
import tensorflow as tf

np.set_printoptions(threshold=np.inf) # display FULL array (infinite)

x=[[2, 2], [3, 3]]
y=[[8, 16], [2, 3]]
#tpow=tf.pow(x, y)  # ==> [[256, 65536], [9, 27]]
tpow=tf.pow(x, 2)  # ==> [[256, 65536], [9, 27]]

sess = tf.Session()
sess.run(tf.initialize_all_variables())
z=sess.run(tpow)
print(z)

