# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np
import re
import time
import copy
from alpr.decorators import memoize, memoize_test_letter, memoize_compute_hog

class TaskSVMLetterDetector(Task):
  def __init__(self, img, hog_descriptor, svm_letters, debug=None):
    self.img=img
    self.hog_descriptor=hog_descriptor
    self.svm_letters=svm_letters
    self.debug=debug
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

    self.search_confusion_map={
      '0O': ['6', '9'],
      '1': ['M', 'K'],
      '2': ['4'],
      '3': [],
      '4': ['2'],
      '5': [],
      '6': [],
      '7': [],
      '8': ['B', 'A'],
      '8B': ['8', 'B'],
      '9': [],
      'A': ['8'],
      'B': [],
      'C': ['E', '8', '8B', 'B'],
      'E': [],
      'H': ['M'],
      'K': ['1'],
      'M': ['H', '1'],
      'P': [],
      'T': [],
      'X': [],
      'Y': [],
      'HM': ['M', 'H']
    }

    self.search_order=['7','9','1','8','0O','8B','HM','2','3','4','5','6','A','B','C','E','H','K','M','P','T','X','Y']
    self.search_yscales=[1.0, 1.2]

  def execute(self):
    img=self.img#[1:25]
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    #guess white color
    white_c=np.median(gray)
    white_c=np.median(gray[gray>white_c])

    window=21
    #simple
    _,th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #adaptive mean
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window, 2)
    #adaptive gaussian
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window, 2)
    #Otsu's
    _,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Otsu's after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(3,3),0)
    _,th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #extend borders
    img=cv2.copyMakeBorder(img, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=(int(white_c), int(white_c), int(white_c)))
    gray=cv2.copyMakeBorder(gray, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=int(white_c))

    th1 = cv2.copyMakeBorder(th1, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=255)
    th2 = cv2.copyMakeBorder(th2, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=255)
    th3 = cv2.copyMakeBorder(th3, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=255)
    th4 = cv2.copyMakeBorder(th4, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=255)
    th5 = cv2.copyMakeBorder(th5, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=255)

    #FIXME destroy russian flag
    #th[22:, 100:]=255
    #th[:, :5]=255
    #th[:, -5:]=255

    self.debug(img, "svm_img")
    self.debug(gray, "svm_gr")
    self.debug(th1, "svm_th1")
    self.debug(th2, "svm_th2")
    self.debug(th3, "svm_th3")
    self.debug(th4, "svm_th4")
    self.debug(th5, "svm_th5")

    #th1_de=th1.copy()
    #th1_de = cv2.dilate(th1_de, kernel, iterations = 1)
    #th1_de = cv2.erode(th1_de, kernel, iterations = 1)
    #th2_de=th2.copy()
    #th2_de = cv2.dilate(th2_de, kernel, iterations = 1)
    #th2_de = cv2.erode(th2_de, kernel, iterations = 1)
    #th3_de=th3.copy()
    #th3_de = cv2.dilate(th3_de, kernel, iterations = 1)
    #th3_de = cv2.erode(th3_de, kernel, iterations = 1)
    #th4_de=th4.copy()
    #th4_de = cv2.dilate(th4_de, kernel, iterations = 1)
    #th4_de = cv2.erode(th4_de, kernel, iterations = 1)
    #th5_de=th5.copy()
    #th5_de = cv2.dilate(th5_de, kernel, iterations = 1)
    #th5_de = cv2.erode(th5_de, kernel, iterations = 1)
    #self.debug(th1_de, "svm_th1de")
    #self.debug(th2_de, "svm_th2de")
    #self.debug(th3_de, "svm_th3de")
    #self.debug(th4_de, "svm_th4de")
    #self.debug(th5_de, "svm_th5de")

    plate=[]

    m=0
    boxes=self.get_boxes_from_contour(th4, gray)+self.get_boxes_from_contour(th3, gray)+self.get_boxes_from_contour(th2, gray)
    #this one should get expanded box
    #boxes+=self.get_boxes_from_contour(th1, gray)
    while len(boxes)>0:
      box=boxes.pop()
      m+=1
      X,Y,W,H, deep, splt = box

      b_img=gray[Y-1:Y+H+1, X-1:X+W+1]

      min_score=1.0
      min_letter=None
      self.debug(b_img, "svm1_t_"+str(m))
      for letter in self.search_order:
        if letter in ['HM', '8B']:
          continue
        score=self.test_letter((X,Y,W,H), gray, letter)
        if score<0:
          self.debug(b_img, "svm1_f_"+letter+"_"+str(m))
        if score<min_score:
          min_score=score
          min_letter=letter

      if min_score<0:
        plate+=[(X+W/2.0, min_letter, (X,Y,W,H), -min_score)]

      if min_score>0:
        for letter in ['8dot', 'dotO', 'dotM', 'dotB', 'dotC']:
          score=self.test_letter((X,Y,W,H), gray, letter)
          if score<0:
            self.debug(b_img, "svm1_f_"+letter+"_"+str(m))
          if score<min_score:
            min_score=score
            if letter=='8dot':
              min_letter='8'
            if letter=='dotO':
              min_letter='0O'
            if letter=='dotM':
              min_letter='M'
            if letter=='dotB':
              min_letter='B'
            if letter=='dotC':
              min_letter='C'
            if letter=='dotH':
              min_letter='H'
            if letter=='dotE':
              min_letter='E'

        if min_score<0:
          plate+=[(X+W/2.0, min_letter, (X,Y,W,H), -min_score)]

      if min_score>0:
        self.debug(b_img, "svm1_nf_"+str(m))

    for box in get_positional_boxes(gray, plate):
      m+=1
      X,Y,W,H, deep = box
      b_img=gray[Y:Y+H,X:X+W]

      fscale=2
      if (X+W/2.0)>img.shape[1]*0.5:
        fscale=2
      if (X+W/2.0)>img.shape[1]*0.75:
        fscale=3
      fscale=30.0*fscale/img.shape[0]
      b_img=cv2.resize(b_img, (0,0), fx=fscale, fy=fscale)
      Yf=int(Y*fscale)
      Xf=int(X*fscale)
      Wf=int(W*fscale)
      Hf=int(H*fscale)

      if Wf<20 or Hf<30:
        continue

      self.debug(b_img, "svm2_"+str(m))

      search_yscales=self.search_yscales
      search_list=copy.copy(self.search_order)
      search_confusion=False

      Found=False

      while len(search_list)>0:
        letter=search_list.pop(0)

        for yscale in search_yscales:
          b_img_=cv2.resize(b_img, (0,0), fx=1.0, fy=yscale) # rescale in case plate contain subline

          found, F=self.hog_descriptor[letter].detectMultiScale(b_img_, 1.05)
          n=0
          for j in xrange(len(found)):
            Found=True
            x,y,w,h=found[j]
            f=F[j][0]

            b_imgl=b_img_[y:y+h,x:x+w]

            if np.abs(yscale-1.0)<0.001:
              self.debug(b_imgl, "svm2_f_"+letter+"_"+str(m)+"_"+str(n))
            else:
              self.debug(b_imgl, "svm2_fs_"+letter+"_"+str(m)+"_"+str(n))

            n+=1

            Xn=X+x/fscale
            Yn=Y+y/fscale/yscale
            Wn=w/fscale
            Hn=h/fscale/yscale
            plate+=[(Xn+Wn/2.0, letter, (Xn,Yn,Wn,Hn), f)]

          if len(found)>0 and not search_confusion and not deep:
            search_confusion=True
            search_list=copy.copy(self.search_confusion_map[letter])
            continue

        deep=True
        if Found and not search_confusion and not deep:
          if yscale!=search_yscales[0]:
            self.search_yscales=[yscale, search_yscales[0]]
          break

    return TaskResultSVMLetterDetector([l[1] for l in sorted(plate)], [l[1:] for l in sorted(plate)])

  def get_boxes_from_contour(self, th, gray, prune=True):
    contours, _ = cv2.findContours(th.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key= lambda x: cv2.contourArea(x), reverse=False)

    plate_area=th.shape[0]*th.shape[1]

    boxes=[]
    m=0
    for cnt in contours:
      m+=1
      X,Y,W,H = cv2.boundingRect(cnt)
      cnt_box_area = W*H

      gray_=gray.copy()
      cv2.drawContours(gray_,[cnt],0,(0,127,127),2)
      self.debug(gray_, "svmcn_"+str(m))

      #select reasonable bounding boxes
      #print cnt_box_area
      if (W+0.0)/H>2.0 or (H+0.0)/W>3.0:
        continue
      if cnt_box_area>plate_area/6 or cnt_box_area<70 and prune:
        continue

      boxes+=[(X,Y,W,H, False, True)]

    return boxes

  #@memoize_test_letter
  def test_letter(self, box, gray, letter):
    svm=self.svm_letters[letter]
    desc=self.compute_hog(box, gray)
    score_=svm.predict(desc, returnDFVal=True)

    return -score_

  #this is runtime potential problem
  @memoize_compute_hog
  def compute_hog(self, box, gray):
    X,Y,W,H=box
    gray=gray[Y-1:Y+H+1, X-1:X+W+1]

    winSize = (20, 30)
    blockSize = (4,6)
    blockStride = (2,3)
    cellSize = (2,3)
    nbins=9

    winStride = (20,30)
    padding = (0,0)

    gray=cv2.resize(gray, winSize, interpolation = cv2.INTER_CUBIC)
    hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
    desc = hog.compute(gray, winStride, padding, ((0, 0),))

    return desc

  def get_positional_boxes(self, th):
    sizes=[(0,25, False),(16,20, False),(27,20, False),(37,20, False), (48,20, False), (59, 20, False), (65,35, True), (85,15, True)]
    boxes=[]
    for X, W, deep in sizes:
      X=int((th.shape[1]-1)*X/100.0)
      W=int((th.shape[1]-1)*W/100.0)
      Y=0
      H=th.shape[0]-1
      boxes+=[(X,Y,W,H, True, None)]

    return boxes

class TaskResultSVMLetterDetector(TaskResult):
  def __init__(self, plate, localization):
    self.plate=plate
    self.localization=localization

def get_positional_boxes(gray, plate):
    mask=np.zeros(gray.shape, np.uint8)
    for p in plate:
      x0, l, box, score=p
      cv2.rectangle(mask, (box[0]+3, box[1]), (box[0]+box[2]-3, box[1]+box[3]), 255, thickness=-1)

    h_gaps=np.mean(mask, 0)
    h_med=np.median(h_gaps[h_gaps>0])
    h_gaps=(h_gaps>h_med/2)+0

    def ranges(gaps):
      lg=len(gaps)
      h1=np.array([i for i in range(lg) if gaps[i]==0 and (i==0 or gaps[i-1]==1)])
      h2=np.array([i for i in range(lg) if gaps[i]==0 and (i==lg-1 or gaps[i+1]==1)])
      gps=[]
      for h in h1:
        gps+=[(h, np.min(h2[h2>=h])+1)]

      return gps

    rgs=[r for r in ranges(h_gaps) if (r[1]-r[0]>5)]
    #print  rgs
    #from matplotlib import pyplot as plt
    #plt.imshow(mask,'gray')
    #plt.show()
    #plt.imshow(gray,'gray')
    #plt.show()

    return [(r[0],0,r[1]-r[0],gray.shape[0], True) for r in rgs]
