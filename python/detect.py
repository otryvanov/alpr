import numpy as np
import cv2
import sys
import os
import re
import alpr

img_name=sys.argv[1]
img = cv2.imread(img_name)

crop=None
transform=None

m = re.match(r'.*2015_08_1[01]_pic_01_.*.jpg', img_name)
if m:
  crop=((450,400), (410,780))
  #angle1=5.3/180*np.pi
  #angle2=99.5/180*np.pi
  #r=1.6
  #transform=np.mat([[np.cos(angle1), r*np.cos(angle2)], [np.sin(angle1), r*np.sin(angle2)]])
  #transform=transform/np.linalg.det(transform)
  #transform=np.linalg.inv(transform)

m = re.match(r'.*2015_08_1[01]_pic_02_.*.jpg', img_name)
if m:
  crop=((350,200), (520,800))

m = re.match(r'.*2015_10_1[0-9]_pic_0[0-1]_.*.jpg', img_name)
if m:
  crop=((300,250), (260,530))
  #angle1=-6.2/180*np.pi
  #angle2=85.5/180*np.pi
  #r=1.1
  #transform=np.mat([[np.cos(angle1), r*np.cos(angle2)], [np.sin(angle1), r*np.sin(angle2)]])
  #transform=transform/np.linalg.det(transform)
  #transform=np.linalg.inv(transform)

datadir=os.path.dirname(os.path.abspath(__file__))
datadir=os.path.join(datadir, 'data')
name=img_name.replace('.jpg','')

engine=alpr.Engine(datadir, crop, transform)
print " ".join(engine.detect(img, name))
sys.stdout.flush()
