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

winSize = (20, 30)
blockSize = (4,6)
blockStride = (2,3)
cellSize = (2,3)
nbins=9

winStride = (20,30)
padding = (0,0)
vectorSize=2916

for f in posfiles:
  img=cv2.imread(f)
  img=cv2.resize(img, winSize, interpolation = cv2.INTER_CUBIC)
  img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  print "Loading positive", f

  for a1 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    for a2 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
      r=1.0
      angle1=a1/180*np.pi
      angle2=np.pi/2+a2/180*np.pi
      transform=np.mat([[np.cos(angle1), r*np.cos(angle2)], [np.sin(angle1), r*np.sin(angle2)]])
      transform=transform/np.linalg.det(transform)
      transform=np.linalg.inv(transform)

      center=(img.shape[1]/2, img.shape[0]/2)
      center=np.mat(center)
      center=np.transpose(center)

      delta=center-np.dot(transform, center)
      transform_mat=np.hstack((transform, delta))

      result=img.copy()
      r_img=cv2.warpAffine(img, transform_mat, (img.shape[1], img.shape[0]), result,
                     cv2.cv.CV_INTER_LINEAR+cv2.cv.CV_WARP_FILL_OUTLIERS,cv2.BORDER_TRANSPARENT)

      #from matplotlib import pyplot as plt
      #plt.imshow(img,'gray')
      #plt.show()

      hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
      desc = hog.compute(r_img, winStride, padding, ((0, 0),))

      descs+=[desc.flatten().tolist()]
      resps+=[1]

for f in negfiles:
  img=cv2.imread(f)
  img=cv2.resize(img, winSize, interpolation = cv2.INTER_CUBIC)
  img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  print "Loading negative", f

  #from matplotlib import pyplot as plt
  #plt.imshow(img,'gray')
  #plt.show()

  hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
  desc = hog.compute(img, winStride, padding)

  descs+=[desc.flatten().tolist()]
  resps+=[2]

descs=np.array(descs, np.float32)
resps=np.array(resps, np.float32)

print "Training with %d samples" % resps.size
print resps

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
