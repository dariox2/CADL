
import numpy as np
import matplotlib.pyplot as plt
import datetime

TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")

np.random.seed(seed=1)

for l in range(3):

  imgs=[]
  fig, axs = plt.subplots(1, 3, figsize=(9, 3))
  for i in range(3):
    newimg=np.trunc(np.random.rand(3,3,3)*255.0)
    print("img: %d = %s" % (i, newimg))
    #np.random.seed(seed=1)
    imgs.append(newimg)
    axs[i].imshow(imgs[i], interpolation="none")
    axs[i].set_title("#%d" % (i))

  plt.savefig("tmp/tnp_"+str(l)+"_"+TID+".png", bbox_inches="tight")

  plt.show()

