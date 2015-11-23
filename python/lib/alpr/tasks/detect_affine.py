# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from alpr.decorators import memoize

class TaskDetectAffine(Task):
  def __init__(self, img, box, debug=None):
    self.img=img
    self.box=box
    self.debug=debug
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

  def execute(self):
    gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    #get single horizontal bias
    h_angle=np.pi/2-horizontal_angle_from_edges(gray)

    #get multiple possible vertical biases
    v_angles=[-np.median(angles)+np.pi/2 for angles in vertical_angles(gray)]

    #debug info
    self.debug(gray, 'lpg')
    ths=thresholded(gray)
    for k in xrange(len(ths)):
      self.debug(ths[k], 'lpg_th'+str(k))

    img_=self.img.copy()
    contours=big_enough_contours(ths[-1])
    for cnt in contours:
      cv2.drawContours(img_,[cnt],0,(0,127,127),2)

    draw_line(img_, (img_.shape[0]/2, img_.shape[1]/2), h_angle)
    for j in xrange(len(v_angles)):
      v_a=v_angles[j]
      color=(0,0,255)
      if j==1:
        color=(0,255,0)
      if j==2:
        color=(255,0,0)
      draw_line(img_, (img_.shape[0]/2, img_.shape[1]/2), v_a, color)

      self.debug(img_, "lpg_afc")

    return TaskResultDetectAffine(self, self.img, [h_angle], v_angles)

class TaskResultDetectAffine(TaskResult):
  def __init__(self, task, img, h_angles, v_angles):
    self.task=task
    self.img=img
    self.h_angles=h_angles
    self.v_angles=v_angles

def draw_line(img, center, theta, color=(0,0,255)):
  cs = np.cos(theta)
  sn = np.sin(theta)
  y0, x0 = center
  x1 = int(x0-100*cs)
  y1 = int(y0+100*sn)
  x2 = int(x0+100*cs)
  y2 = int(y0-100*sn)
  cv2.line(img, (x1,y1), (x2,y2), color, 2)

@memoize
def hough_horizontal_angles(img, n=3):
  edges = cv2.Canny(img, 50, 150, apertureSize = 3)

  angle_precision=np.pi/180/4
  lines = cv2.HoughLines(edges, 1, angle_precision, 3)

  if lines is None:
    return []

  thetas=np.array(lines[0])[:,1]

  #filter in first(assumed to be good horizontal) and nearest
  thetas=thetas[(np.abs(thetas-thetas[0])<np.pi/12)]

  return list(thetas[0:n])

@memoize
def hough_vertical_angles(img, n=5):
  edges = cv2.Canny(img, 50, 150, apertureSize = 3)

  angle_precision=np.pi/180/4
  lines = cv2.HoughLines(edges, 1, angle_precision, 3)

  if lines is None:
    return []

  thetas=np.array(lines[0])[:,1]

  #filter in nearest to vertical
  thetas=thetas[np.abs(thetas - np.pi/2) > np.pi/3]
  thetas=thetas-np.pi*(thetas > np.pi/2)

  return list(thetas[0:n])

def thresholded(gray):
  window=21 #magic number

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

  return [th1, th2, th3, th4 ,th5]

def horizontal_angle_from_edges(gray):
  #detect horizontal in different splits to compensate for car decorations
  #use different binarization for stability
  #FIXME should use this data to detect bend plates
  bands=[(0.5, 1, 0.5, 1), (0.5, 1, 0, 0.5), (0, 0.5, 0.5, 1), (0, 0.5, 0, 0.5), (0, 1, 0.25, 0.75)]
  thetas_b=[]
  for b in bands:
    thetas_b+=[hough_horizontal_angles(i[int(i.shape[0]*b[0]):int(i.shape[0]*b[1]),int(i.shape[1]*b[2]):int(i.shape[1]*b[3])]) for i in [gray]+thresholded(gray)[1:3]]

  thetas=[]
  for l in thetas_b:
    thetas+=l
  thetas=np.array(thetas).flatten()

  theta_avg=np.median(thetas) #check median vs mean
  theta_std=np.std(thetas)

  #finter out uncommon value
  thetas=thetas[np.abs(thetas-theta_avg)<theta_std]

  theta=np.median(thetas) #check median vs mean
  return theta

def vertical_angles_from_edges(gray):
  #detect vertical lines from left and right part
  #band 1,2 catch plate horizontal border, semi vertical lines in letters _and_ car decorations
  #band 3 catches semi vertical lines in letters
  bands=[(0, 1, 0.5, 1), (0, 1, 0, 0.5), (0, 1, 0.25, 0.75)]
  thetas_b=[]
  for b in bands:
    thetas_b+=[hough_vertical_angles(i[int(i.shape[0]*b[0]):int(i.shape[0]*b[1]),int(i.shape[1]*b[2]):int(i.shape[1]*b[3])]) for i in [gray]+thresholded(gray)[1:3]]

  thetas=[]
  for l in thetas_b:
    thetas+=l
  thetas=np.array(thetas).flatten()

  #probably should whiten thetas
  #3 clusters chosen because in general they corresponds to vertical and almost vertical lines in symbols
  centroids,_ = kmeans(thetas, 3)
  idx,_ = vq(thetas,centroids)

  data=[list(thetas[idx==i]) for i in xrange(3)]
  data=sorted([d for d in data if d]) #sort before merge

  changed=True
  while changed:
    changed=False
    for i in xrange(0, len(data)-1):
      # merge if less than 5 degree away
      if abs(np.median(data[i])-np.median(data[i+1]))<0.087:
        data[i]+=data[i+1]
        data.pop(i+1)
        changed=True
        break

  return data

def vertical_angles_from_contours(img):
  contours=big_enough_contours(img)

  thetas=[]
  for cnt in contours:
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    #upscale to get better precision
    mask=cv2.resize(mask,(0,0), fx=2.0, fy=2.0)
    thetas+=hough_vertical_angles(mask[:,int(mask.shape[1]*0.5):int(mask.shape[1]*1)], 5)
    thetas+=hough_vertical_angles(mask[:,int(mask.shape[1]*0):int(mask.shape[1]*0.5)], 5)

  thetas=np.array(thetas).flatten()

  #probably should whiten thetas
  #3 clusters chosen because in general they corresponds to vertical and almost vertical lines in symbols
  centroids,_ = kmeans(thetas, 3)
  idx,_ = vq(thetas,centroids)

  data=[list(thetas[idx==i]) for i in xrange(3)]
  data=sorted([d for d in data if d]) #sort before merge

  changed=True
  while changed:
    changed=False
    for i in xrange(0, len(data)-1):
      # merge if less than 5 degree away
      if abs(np.median(data[i])-np.median(data[i+1]))<0.087:
        data[i]+=data[i+1]
        data.pop(i+1)
        changed=True
        break

  return data

def vertical_angles(img):
  ths=thresholded(img)

  thetas2=vertical_angles_from_edges(img)
  thetas1=vertical_angles_from_contours(ths[-1])

  thetas1_=[np.median(c) for c in thetas1]
  thetas2_=[np.median(c) for c in thetas2]

  #get consistent angles
  th=[]
  for theta2 in thetas2_:
    if np.min(np.abs(theta2-thetas1_))<0.1:
      th+=[theta2]

  if len(th)==1:
    return [th[0]]
  elif len(th)>1:
    return sorted(th, reverse=False, key=lambda x: np.abs(x)) #FIXME should replace with sensible bias
  else:
    return sorted(thetas2_, reverse=False, key=lambda x: np.abs(x))

def big_enough_contours(img):
  big_size=img.shape[0]*img.shape[1]/20 #magic number

  img_=img.copy() #because next operation is destructive
  contours, _ = cv2.findContours(img_, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours=[cnt for cnt in contours if big_size < cv2.contourArea(cnt)]

  return contours
