print("Imports...")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as ipyd
from libs import gif, nb_utils
sess = tf.InteractiveSession()
from libs import inception
print("load inception model...")
net = inception.get_inception_model()
print("graph_def...")
print(net['graph_def'])

