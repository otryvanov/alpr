import xml.etree.ElementTree as ET
import numpy as np
import cv2
import sys
import re
import pickle
from os import listdir
from os.path import isfile, join

posdir=sys.argv[1]
negdir=sys.argv[2]

posfiles = [join(posdir,f) for f in listdir(posdir) if isfile(join(posdir,f))]
negfiles = [join(negdir,f) for f in listdir(negdir) if isfile(join(negdir,f))]

descs=[]
resps=[]

winSize = (180, 60)
blockSize = (12,12)
blockStride = (12,12)
cellSize = (6,6)
nbins=9

winStride = (180,60)
padding = (0,0)
vectorSize=2700

for f in posfiles:
  img=cv2.imread(f)
  img=cv2.resize(img, winSize, interpolation = cv2.INTER_CUBIC)
  img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  imgs=[img, cv2.flip(img,1), cv2.flip(img,0), cv2.flip(cv2.flip(img,1),0)]
  for im in imgs:
    hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
    desc = hog.compute(im, winStride, padding, ((0, 0),))

    descs+=[desc.flatten().tolist()]
    resps+=[1]

for f in negfiles:
  img=cv2.imread(f)
  img=cv2.resize(img, winSize, interpolation = cv2.INTER_CUBIC)
  img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  imgs=[img, cv2.flip(img,1), cv2.flip(img,0), cv2.flip(cv2.flip(img,1),0)]
  for im in imgs:
    hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
    desc = hog.compute(im, winStride, padding, ((0, 0),))

    descs+=[desc.flatten().tolist()]
    resps+=[-1]

descs=np.array(descs, np.float32)
resps=np.array(resps, np.float32)

svm = cv2.SVM()
svm_params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 1)
svm.train_auto(np.array(descs), np.array(resps), None, None, params=svm_params, k_fold=5)

svm.save("svm.xml")

tree = ET.parse('svm.xml')
root = tree.getroot()
SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
svmvec.append(-rho)
pickle.dump(svmvec, open("svm.pickle", 'w'))
