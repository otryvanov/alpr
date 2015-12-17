# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import os
import re
import xml.etree.ElementTree as et
from alpr.tasks.zone_transform import *
from alpr.tasks.haar_localization import *
from alpr.tasks.horizontal_deskew import *
from alpr.tasks.vertical_deskew import *
from alpr.tasks.fine_crop import *
from alpr.tasks.svm_letter_detector import *
from alpr.tasks.merge_plate import *

class Engine:
  def __init__(self, datadir, crop=None, transform=None):
    self.datadir=datadir
    self.crop=crop #has form of tuple of ((y,x),(height, width))
    self.transform=transform #has form of np.mat with shape (2,2)
    self.haar_cascade = cv2.CascadeClassifier(os.path.join(datadir,'haarcascade_russian_plate_number.xml'))
    self.hog_descriptor={}
    self.letter_svm={}

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

    for letter in ['HM','0O','1','2','3','4','5','6','7','8','8B','9','A','B','C','E','H','K','M','P','T','X','Y','8dot', 'dotH', 'dotM', 'dotO', 'dotE', 'dotC', 'dotP', 'dotB']:
      svm_file=os.path.join(datadir,'svm_letter_'+letter+'_stage_1.xml')
      self.letter_svm[letter]=cv2.SVM()
      self.letter_svm[letter].load(svm_file)

    svm_file=os.path.join(datadir,'svm_plate.xml')
    self.plate_svm=cv2.SVM()
    self.plate_svm.load(svm_file)

    svm_file=os.path.join(datadir,'svm_vertical_stage1.xml')
    self.vertical_stage1_svm=cv2.SVM()
    self.vertical_stage1_svm.load(svm_file)

  def detect(self, img, name):
    debug_var={"images": [img], "titles": ['orig'], "suffixes": ['']}

    transformed=None

    def debug(img, name):
      debug_var["images"]+=[img]
      debug_var["titles"]+=[name]
      debug_var["suffixes"]+=[name]

    queue=[TaskZoneTransform(img, debug, self.crop, self.transform)]
    plates=[]
    while len(queue)>0:
      task=queue.pop(0)
      #print "Executing "+str(task)
      result=task.execute()
      if isinstance(result, TaskResultZoneTransform):
        transformed=result.img
        queue+=[TaskHaarLocalization(result.img, self.haar_cascade, self.plate_svm, 0.1, debug)]
      elif isinstance(result, list) and len(result)>0 and isinstance(result[0], TaskResultHaarLocalization):
        for n in xrange(len(result[:1])):
          #print result[n].score
          if result[n].score>0:
            debug(result[n].img, 'haar_bad_'+str(n))
          else:
            debug(result[n].img, 'haar_'+str(n))

        queue+=[TaskHorizontalDeskew(r.img, transformed, r.box, debug) for r in result[:1]]
      elif isinstance(result, TaskResultHorizontalDeskew):
        r=result
        queue+=[TaskVerticalDeskew(r.img, r.parent, r.box, debug)]
      elif isinstance(result, list) and len(result)>0 and isinstance(result[0], TaskResultVerticalDeskew):
        r_imgs=[]
        for r in result:
          winSize = (180, 60)
          blockSize = (12,12)
          blockStride = (12,12)
          cellSize = (6,6)
          nbins=18

          winStride = (180,60)
          padding = (0,0)

          r_img=cv2.resize(r.img, winSize, interpolation = cv2.INTER_CUBIC)
          gray=cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
          hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
          desc = hog.compute(gray, winStride, padding, ((0, 0),))
          score_=self.vertical_stage1_svm.predict(desc, returnDFVal=True)

          r_imgs+=[(score_, r.img)]
        n=0
        for score_,rotated_img in sorted(r_imgs, key=lambda x: x[0])[:1]:
          debug(rotated_img, 'lpv2_'+str(n))
          n+=1
          queue+=[TaskFineCrop(rotated_img, debug)]
      elif isinstance(result, list) and len(result)>0 and isinstance(result[0], TaskResultFineCrop):
        queue+=[TaskSVMLetterDetector(r.img, self.hog_descriptor, self.letter_svm, debug) for r in result]
      elif isinstance(result, TaskResultSVMLetterDetector):
        queue+=[TaskMergePlate([result.localization])]
      elif isinstance(result, TaskResultMergePlate):
        plates+=[result.plate]

    #return plates
    #
    #from matplotlib import pyplot as plt
    #size=8
    #for i in xrange(len(debug_var["images"])):
    #  plt.subplot(int(len(debug_var["images"])/(size+0.0)+1),size,i+1),plt.imshow(debug_var["images"][i],'gray')
    #  plt.title(debug_var["titles"][i])
    #  plt.xticks([]),plt.yticks([])
    #plt.show()
    #return plates

    #for i in xrange(1,len(debug_var["images"])):
    #    cv2.imwrite(name+"_"+debug_var["suffixes"][i]+".jpg",debug_var["images"][i])

    return plates
