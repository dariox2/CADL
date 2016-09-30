
import os
import numpy as np
import matplotlib.pyplot as plt

def getfilelist(path='./'):
  fs = [os.path.join(path, f)
  for f in os.listdir(path) if f.endswith('.jpg')]
  fs=sorted(fs)
  return fs


def convert(fnam):
  img=plt.imread(fnam)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      for k in range(img.shape[2]):
        img[i,j,k]=img[i,j,2]

  plt.imsave(arr=img, fname=fnam,)


#
# main
#

fl=getfilelist()

for f in fl:
  print(f)
  convert(f)

# eop 


