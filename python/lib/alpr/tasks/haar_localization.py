# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np

class TaskHaarLocalization(Task):
  def __init__(self, img, cascade, svm_scorer, padding, debug=None):
    self.img=img
    self.cascade=cascade
    self.padding=padding
    self.svm_scorer=svm_scorer
    self.debug=debug
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

  def execute(self):
    gray=cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    min_neighbours=5
    resize=1.1
    result=self.localize(gray, resize, min_neighbours)
    if len(result)>0:
      return sorted(result, key = lambda x: x.score)

    #dig deeper with lower precision
    #blur out weak lattice structures
    resize=1.05
    params=[(1,5,1), (1,7,1), (3,5,1), (3,7,1)]
    for w,h,min_neighbours in params:
      scores=[]
      min_score=1.0

      gray_=cv2.GaussianBlur(gray, (w,h), 0)
      result+=self.localize(gray_, resize, min_neighbours)

      if len(result)>0:
        min_score=min([r.score for r in result])
      if min_score<=-1.0:
        break

    return sorted(result, key = lambda x: x.score)

  def localize(self, gray, resize, min_neighbours):
    lps=self.cascade.detectMultiScale(gray, resize, min_neighbours)

    result=[]
    for (x,y,w,h) in lps:
      padding=self.get_padding(x,y,w,h)
      x=int(x-w*padding)
      y=int(y-h*padding)
      w=int(w*(1+2.0*self.padding))
      h=int(h*(1+2.0*self.padding))

      lp=self.img[y:y+h,x:x+w]
      self.debug(lp, 'lp')
      result+=[TaskResultHaarLocalization(lp, (x,y,w,h), self.score(lp))]

    return result

  def get_padding(self, x,y,w,h):
    padding=self.padding
    if x-w*padding<0:
      padding=(x+0.0)/w
    if y-h*padding<0:
      padding=(y+0.0)/h
    if x+w*padding > self.img.shape[1]:
      padding=(self.img.shape[1]-x+0.0)/w
    if y+h*padding > self.img.shape[0]:
      padding=(self.img.shape[0]-y+0.0)/h

    return padding

  def score(self, img):
    winSize = (180, 60)
    blockSize = (12,12)
    blockStride = (12,12)
    cellSize = (6,6)
    nbins=9

    winStride = (180,60)
    padding = (0,0)
    vectorSize=2700

    img=cv2.resize(img, winSize, interpolation = cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
    desc = hog.compute(gray, winStride, padding, ((0, 0),))
    score_=self.svm_scorer.predict(desc, returnDFVal=True)

    return score_

class TaskResultHaarLocalization(TaskResult):
  def __init__(self, img, box, score):
    self.img=img
    self.box=box
    self.score=score
