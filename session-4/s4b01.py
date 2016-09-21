
print("Begin...")
import numpy as np
import matplotlib.pyplot as plt

print("Loading tensorflow...")
import tensorflow as tf

#import IPython.display as ipyd
from libs import gif, nb_utils

sess = tf.InteractiveSession()

from libs import inception
print("Loading inception...")
net = inception.get_inception_model()

nb_utils.show_graph(net['graph_def'])

tf.import_graph_def(net['graph_def'], name='inception')

print(net['labels'])

g = tf.get_default_graph()
names = [op.name for op in g.get_operations()]
print(names)
