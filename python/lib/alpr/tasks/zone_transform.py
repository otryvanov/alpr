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
    result=zone_transform(self.img, self.crop, self.transform)[1]

    self.debug(result, 'trmd')

    return TaskResultZoneTransform(result)

class TaskResultZoneTransform(TaskResult):
  def __init__(self, img):
    self.img=img

def zone_transform(img, crop=None, transform=None):
  result=img.copy()
  if transform is not None:
    center=(img.shape[1]/2, img.shape[0]/2)
    if crop:
      center=[int(crop[0][1]+crop[1][1]/2.0), int(crop[0][0]+crop[1][0]/2.0)]
    center=np.mat(center)
    center=np.transpose(center)
    delta=center-np.dot(transform, center)
    transform_mat=np.hstack((transform, delta))

    cv2.warpAffine(img, transform_mat, (img.shape[1], img.shape[0]), result,
                   cv2.cv.CV_INTER_LINEAR+cv2.cv.CV_WARP_FILL_OUTLIERS,cv2.BORDER_TRANSPARENT)

  result_crpd=result
  if crop is not None:
    result_crpd=result[crop[0][0]:crop[0][0]+crop[1][0], crop[0][1]:crop[0][1]+crop[1][1]]

  return result, result_crpd
