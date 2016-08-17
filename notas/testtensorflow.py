#
# Test instalacion TensorFlow + GTK
#
#OK, PROBADO EN AILINUX (UBUNTU) Y CENTOS 
#


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

sess=tf.InteractiveSession()

x=tf.linspace(-3.0, 3.0, 100)

mean=0
sigma=1.0
z=(tf.exp(tf.neg(tf.pow(x-mean, 2.0)/(2.0*tf.pow(sigma, 2.0))))*(1.0/(sigma*tf.sqrt(2.0*3.1415))))

res=z.eval()
plt.plot(res)
plt.show()

# eop

