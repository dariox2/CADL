
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

#plt.style.use('ggplot')

from libs import utils
files = utils.get_celeb_files()

img = plt.imread(files[50])
plt.imshow(img)
plt.show()

##import time
##time.sleep(5)

