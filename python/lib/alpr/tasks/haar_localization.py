# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np

class TaskHaarLocalization(Task):
  def __init__(self, img, cascade, padding, debug=None):
    self.img=img
    self.cascade=cascade
    self.padding=padding
    self.debug=debug
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

  def execute(self):
    gray=cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    lps=self.cascade.detectMultiScale(gray, 1.1, 5)
    result=[]

    for (x,y,w,h) in lps:
      padding=self.get_padding(x,y,w,h)
      x=int(x-w*padding)
      y=int(y-h*padding)
      w=int(w*(1+2.0*self.padding))
      h=int(h*(1+2.0*self.padding))

      lp=self.img[y:y+h,x:x+w]
      self.debug(lp, 'lp')
      result+=[TaskResultHaarLocalization(lp, (x,y,w,h))]

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

class TaskResultHaarLocalization(TaskResult):
  def __init__(self, img, box):
    self.img=img
    self.box=box
