# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np

class TaskZoneTransform(Task):
  def __init__(self, img, debug=None, crop=None, transform=None):
    self.img=img
    self.crop=crop
    self.transform=transform
    self.debug=debug
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

  def execute(self):
    result=self.img.copy()
    if self.transform is not None:
      center=(self.img.shape[1]/2, self.img.shape[0]/2)
      if self.crop:
        center=[int(self.crop[0][1]+self.crop[1][1]/2.0), int(self.crop[0][0]+self.crop[1][0]/2.0)]
      center=np.mat(center)
      center=np.transpose(center)

      delta=center-np.dot(self.transform, center)
      transform_mat=np.hstack((self.transform, delta))

      cv2.warpAffine(self.img, transform_mat, (self.img.shape[1], self.img.shape[0]), result,
                     cv2.cv.CV_INTER_LINEAR+cv2.cv.CV_WARP_FILL_OUTLIERS,cv2.BORDER_TRANSPARENT)

    if self.crop is not None:
      result=result[self.crop[0][0]:self.crop[0][0]+self.crop[1][0], self.crop[0][1]:self.crop[0][1]+self.crop[1][1]]

    self.debug(result, 'trmd')

    return TaskResultZoneTransform(result)

class TaskResultZoneTransform(TaskResult):
  def __init__(self, img):
    self.img=img
