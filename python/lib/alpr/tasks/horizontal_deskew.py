# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
from alpr.tasks.zone_transform import *
import cv2
import numpy as np
from alpr.decorators import memoize

class TaskHorizontalDeskew(Task):
  def __init__(self, img, parent, box, debug=None):
    self.img=img
    self.parent=parent
    self.box=box
    self.debug=debug
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

  def execute(self):
    gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,8))
    gray = clahe.apply(gray)

    crop=self.box
    crop=((crop[1],crop[0]), (crop[3],crop[2]))

    #debug info
    self.debug(gray, 'lph0g')
    ths=thresholded(gray)
    for k in xrange(len(ths)):
      self.debug(ths[k], 'lph0th'+str(k))

    #get single horizontal bias and transform img
    h_angle=np.pi/2-horizontal_angle_from_edges(gray, np.pi/12, np.pi/180)
    d_parent, d_img = deskew_from_parent(self.parent, crop, h_angle)
    self.debug(d_img, 'lph1')

    return TaskResultHorizontalDeskew(d_img, d_parent, self.box, h_angle)

class TaskResultHorizontalDeskew(TaskResult):
  def __init__(self, img, parent, box, h_angle):
    self.img=img
    self.parent=parent
    self.box=box
    self.h_angle=h_angle

@memoize
def hough_horizontal_angles(img, band=None, n=3, cut=np.pi/12, precision=np.pi/180):
  edges = cv2.Canny(img, 50, 150, apertureSize = 3, L2gradient=True)

  if band is not None:
    edges=edges[int(edges.shape[0]*band[0]):int(edges.shape[0]*band[1]),int(edges.shape[1]*band[2]):int(edges.shape[1]*band[3])]

  lines = cv2.HoughLines(edges, 1, precision, 3)

  if lines is None:
    return []

  thetas=np.array(lines[0])[:,1]

  #filter in first(assumed to be good horizontal) and nearest
  thetas=thetas[(np.abs(thetas-thetas[0])<cut)]

  return list(thetas[0:n])

def thresholded(gray):
  window=21 #magic number

  #simple
  _,th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  #adaptive mean
  blur = cv2.GaussianBlur(gray,(3,3),0)
  th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window, 2)
  #adaptive gaussian
  blur = cv2.GaussianBlur(gray,(3,3),0)
  th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window, 2)
  #Otsu's
  _,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #Otsu's after Gaussian filtering
  blur = cv2.GaussianBlur(gray,(3,3),0)
  _,th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  return [th1, th2, th3, th4 ,th5]

def horizontal_angle_from_edges(gray, cut=np.pi/12, precision=np.pi/180):
  #detect horizontal in different splits to compensate for car decorations
  #use different binarization for stability
  #FIXME should use this data to detect bend plates

  #we do not care about bands since we retry later with better plate cut
  #bands=[(0, 1, 0, 0.5), (0, 1, 0.5, 1), (0, 1, 0.25, 0.75), (0, 1, 0, 1)]
  bands=[(0, 0.5, 0, 0.5), (0.5, 1, 0.5, 1),  (0, 0.5, 0.5, 1), (0.5, 1, 0, 0.5), (0, 0.5, 0.25, 0.75), (0.5, 1, 0.25, 0.75)]

  thetas_b=[]
  for b in bands:
    ths=thresholded(gray)
    thetas_b+=[hough_horizontal_angles(i, band=b, n=3, cut=cut, precision=precision) for i in [gray, ths[2], ths[4]]]

  thetas=[]
  map(thetas.extend, thetas_b)
  thetas=np.array(thetas).flatten()

  theta_avg=np.median(thetas) #check median vs mean
  theta_std=np.std(thetas)

  while theta_std>np.pi/180:
    #filter out uncommon values
    thetas=thetas[np.abs(thetas-theta_avg)<=theta_std+0.00001]
    theta_avg=np.median(thetas) #check median vs mean
    theta_std=np.std(thetas)

  theta=np.mean(thetas) #check median vs mean
  return theta

def deskew_from_parent(img, crop, h_angle):
  transform=np.mat([[1, 0], [np.tan(-h_angle), 1]])
  transform=np.linalg.inv(transform)

  d_img, d_crop=zone_transform(img, crop, transform)

  return (d_img, d_crop)
