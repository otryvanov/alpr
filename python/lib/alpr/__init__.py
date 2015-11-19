# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import os
import xml.etree.ElementTree as et
from alpr.tasks.zone_transform import *
from alpr.tasks.haar_localization import *
from alpr.tasks.detect_affine import *
from alpr.tasks.fine_crop import *
from alpr.tasks.svm_letter_detector import *
from alpr.tasks.merge_plate import *

class Engine:
  def __init__(self, img, name, datadir, crop=None, transform=None):
    self.img=img
    self.datadir=datadir
    self.crop=crop #has form of tuple of ((y,x),(height, width))
    self.transform=transform #has form of np.mat with shape (2,2)
    self.name=name
    self.haar_cascade = cv2.CascadeClassifier(os.path.join(datadir,'haarcascade_russian_plate_number.xml'))
    self.hog_descriptor={}

    winSize = (20, 30)
    blockSize = (4,6)
    blockStride = (2,3)
    cellSize = (2,3)
    nbins=9

    winStride = (20,30)
    padding = (0,0)

    for letter in ['HM','0O','1','2','3','4','5','6','7','8','8B','9','A','B','C','E','H','K','M','P','T','X','Y']:
      self.hog_descriptor[letter]=cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

      svm_file=os.path.join(datadir,'svm_letter_'+letter+'.xml')
      tree = et.parse(svm_file)
      root = tree.getroot()
      # nasty hacks not described in any docs
      svs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
      rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
      svmvec = [float(x) for x in re.sub( '\s+', ' ', svs.text ).strip().split(' ')]+[-rho]
      self.hog_descriptor[letter].setSVMDetector(np.array(svmvec))

  def detect(self):
    debug_var={"images": [self.img], "titles": ['orig'], "suffixes": ['']}

    transformed=None

    def debug(img, name):
      debug_var["images"]+=[img]
      debug_var["titles"]+=[name]
      debug_var["suffixes"]+=[name]

    queue=[TaskZoneTransform(self.img, debug, self.crop, self.transform)]
    plates=[]
    while len(queue)>0:
      task=queue.pop(0)
      #print "Executing "+str(task)
      result=task.execute()
      if isinstance(result, TaskResultZoneTransform):
        transformed=result.img
        queue+=[TaskHaarLocalization(result.img, self.haar_cascade, 0.1, debug)]
      elif isinstance(result, list) and len(result)>0 and isinstance(result[0], TaskResultHaarLocalization):
        queue+=[TaskDetectAffine(r.img, r.box, debug) for r in result]
      elif isinstance(result, TaskResultDetectAffine):
        #FIXME this is wrong place to do this
        #but we need to access prelocalized image which is not avaliable in TaskDetectAffine
        img=result.img
        angles1=result.h_angles
        angles2=result.v_angles
        angle1=angles1[0]

        for a1 in angles1:
          for a2 in angles2[:1]:
            angle1=-a1#-np.pi/180/10
            angle2=np.pi-a2
            r=1
            transform=np.mat([[np.cos(angle1), r*np.cos(angle2)], [np.sin(angle1), r*np.sin(angle2)]])
            transform=transform/np.linalg.det(transform)
            transform=np.linalg.inv(transform)

            crop=result.task.box
            crop=((crop[1],crop[0]), (crop[3],crop[2]))
            task=TaskZoneTransform(transformed, None, crop, transform)
            rotated_img=task.execute().img
            debug(rotated_img, 'aft')
            queue+=[TaskFineCrop(rotated_img, debug)]
      elif isinstance(result, list) and len(result)>0 and isinstance(result[0], TaskResultFineCrop):
        queue+=[TaskSVMLetterDetector(r.img, self.hog_descriptor, debug) for r in result]
      elif isinstance(result, TaskResultSVMLetterDetector):
        queue+=[TaskMergePlate([result.localization])]
        #print result.plate, result.localization
      elif isinstance(result, TaskResultMergePlate):
        plates+=[result.plate]

    return plates

    from matplotlib import pyplot as plt
    for i in xrange(len(debug_var["images"])):
      plt.subplot(int(len(debug_var["images"])/6.0+1),6,i+1),plt.imshow(debug_var["images"][i],'gray')
      plt.title(debug_var["titles"][i])
      plt.xticks([]),plt.yticks([])
    plt.show()

    return plates

    for i in xrange(1,len(debug_var["images"])):
      cv2.imwrite(self.name+"_"+debug_var["suffixes"][i]+".jpg",debug_var["images"][i])
