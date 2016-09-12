#
# count pixels
#

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('puntos.jpg')
#plt.imshow(img)
#plt.show()

#tp=0
pn=0
for i in range(0, img.shape[0]):
  for j in range(0, img.shape[1]):
    #print(i,j,img[i,j])
    #tp+=1
    if (img[i,j]==[0,0,0]).all():
      pn += 1
     
print("puntos negros: ", pn)



