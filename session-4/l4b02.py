
#
# Session 4 part 3: Style Net
#

#
# Style Net
#

# Leon Gatys and his co-authors demonstrated a pretty epic extension 
# to deep dream which showed that neural networks trained on objects 
# like the one we've been using actually represent both content and 
# style, and that these can be independently manipulated, for 
# instance taking the content from one image, and the style from 
# another. They showed how you could artistically stylize the same 
# image with a wide range of different painterly aesthetics. Let's 
# take a look at how we can do that. We're going to use the same 
# network that they've used in their paper, VGG. This network is a 
# lot less complicated than the Inception network, but at the 
# expense of having a lot more parameters.

#
# VGG Network
#

# In the resources section, you can find the library for loading 
# this network, just like you've done w/ the Inception network. 
# Let's reset the graph:

print("Loading tensorflow...")
import tensorflow as tf
from libs import utils

#from tensorflow.python.framework.ops import reset_default_graph
#sess.close()
#reset_default_graph()

# dja
#import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
import datetime
#np.set_printoptions(threshold=np.inf) # display FULL array (infinite)
plt.ion()
plt.figure(figsize=(5, 5))
TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")


# And now we'll load up the new network, except unlike before, we're 
# going to explicitly create a graph, and tell the session to use 
# this graph. If we didn't do this, tensorflow would just use the 
# default graph that is always there. But since we're going to be 
# making a few graphs, we'll need to do it like this.

# OJO! 500 MB
from libs import vgg16
print("DOWNLOADING VGG16")
net = vgg16.get_vgg_model()

# Note: We will explicitly define a context manager here to handle 
# the graph and place the graph in CPU memory instead of GPU memory, 
# as this is a very large network!

g = tf.Graph()
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    names = [op.name for op in g.get_operations()]

# Let's take a look at the network:

# REQUIRES TENSORBOARD
# nb_utils.show_graph(net['graph_def'])

print(names)










input("End")

# eop

