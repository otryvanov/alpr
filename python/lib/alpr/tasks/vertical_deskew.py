# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
from alpr.tasks.zone_transform import *
from alpr.tasks.horizontal_deskew import thresholded
import cv2
import numpy as np

class TaskVerticalDeskew(Task):
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
    min_y=int(self.img.shape[0]*0.15)
    max_y=int(self.img.shape[0]*0.95)
    min_x=int(self.img.shape[1]*0.05)
    max_x=int(self.img.shape[1]*0.95)
    img_c=self.img[min_y:max_y,min_x:max_x]
    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,8))
    gray = clahe.apply(gray)

    crop=self.box
    crop=((crop[1],crop[0]), (crop[3],crop[2]))

    #debug info
    self.debug(gray, 'lpv0g')
    ths=thresholded(gray)
    for k in xrange(len(ths)):
      self.debug(ths[k], 'lpv0th'+str(k))

    result=[]

    #get multiple possible vertical biases
    v_angles=[-np.median(angles)+np.pi/2 for angles in vertical_angles(gray)]
    for v_angle in v_angles:
      d_parent, d_img = deskew_from_parent(self.parent, crop, v_angle)
      self.debug(d_img, 'lpv1')
      result+=[TaskResultVerticalDeskew(d_img, d_parent, self.box, v_angle)]

    #debug info
    img_=img_c.copy()
    contours=big_enough_contours(ths[-1])
    for cnt in contours:
      cv2.drawContours(img_,[cnt],0,(0,127,127),2)

    for j in xrange(len(v_angles)):
      v_a=v_angles[j]
      color=(0,0,255)
      if j==1:
        color=(0,255,0)
      if j==2:
        color=(255,0,0)
      draw_line(img_, (img_.shape[0]/2, img_.shape[1]/2), v_a, color)

      self.debug(img_, "lpv1l")

    return result

class TaskResultVerticalDeskew(TaskResult):
  def __init__(self, img, parent, box, v_angle):
    self.img=img
    self.parent=parent
    self.box=box
    self.v_angle=v_angle

def deskew_from_parent(img, crop, v_angle):
  v_angle=np.pi-v_angle

  transform=np.mat([[1, -np.cos(v_angle)], [0, 1]])
  d_img, d_crop=zone_transform(img, crop, transform)

  return (d_img, d_crop)

def draw_line(img, center, theta, color=(0,0,255)):
  cs = np.cos(theta)
  sn = np.sin(theta)
  y0, x0 = center
  x1 = int(x0-100*cs)
  y1 = int(y0+100*sn)
  x2 = int(x0+100*cs)
  y2 = int(y0-100*sn)
  cv2.line(img, (x1,y1), (x2,y2), color, 2)

def hough_vertical_angles(img, band=None, n=5, cut=np.pi/3, precision=np.pi/180/4):
  edges = cv2.Canny(img, 50, 150, apertureSize = 3, L2gradient=True)

  if band is not None:
    edges=edges[int(edges.shape[0]*band[0]):int(edges.shape[0]*band[1]),int(edges.shape[1]*band[2]):int(edges.shape[1]*band[3])]

  lines = cv2.HoughLines(edges, 1, precision, 3)

  if lines is None:
    return []

  thetas=np.array(lines[0])[:,1]

  #filter in nearest to vertical
  thetas=thetas[np.abs(thetas - np.pi/2) > cut]
  thetas=thetas-np.pi*(thetas > np.pi/2)

  return list(thetas[0:n])

def vertical_angles_from_edges(gray):
  #detect vertical lines from left and right part
  #band 1,2 catch plate horizontal border, semi vertical lines in letters _and_ car decorations
  #band 3 catches semi vertical lines in letters
  bands=[(0, 1, 0.5, 1), (0, 1, 0, 0.5), (0, 1, 0.25, 0.75)]
  thetas_b=[]
  for b in bands:
    thetas_b+=[hough_vertical_angles(i, band=b) for i in [gray]+thresholded(gray)[1:3]]

  thetas=[]
  map(thetas.extend, thetas_b)
  thetas=np.array(thetas).flatten()

  if len(thetas)==0:
    return []

  #probably should whiten thetas
  #4 clusters chosen because in general they corresponds to vertical and almost vertical lines in symbols
  n_cl=min(4, len(thetas))

  thetas=np.float32(thetas)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
  flags = cv2.KMEANS_PP_CENTERS
  compactness,idx,centers = cv2.kmeans(thetas,n_cl,criteria,10,flags)
  idx=idx.flatten()

  data=[list(thetas[idx==i]) for i in xrange(n_cl)]
  data=sorted([d for d in data if d]) #sort before merge

  changed=True
  while changed:
    changed=False
    for i in xrange(0, len(data)-1):
      #merge if less than 5 degree away
      if abs(np.median(data[i])-np.median(data[i+1]))<0.17453/2:
        data[i]+=data[i+1]
        data.pop(i+1)
        changed=True
        break

  return data

def vertical_angles_from_contours(img):
  contours=big_enough_contours(img)

  thetas=[]
  thetas_b=[]
  bands=[(0, 1, 0.5, 1), (0, 1, 0, 0.5)]
  for cnt in contours:
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    for b in bands:
      thetas_b+=[hough_vertical_angles(mask, band=b, n=5)]#, n=2)]

  thetas=[]
  map(thetas.extend, thetas_b)
  thetas=np.array(thetas).flatten()

  if len(thetas)==0:
    return []

  #probably should whiten thetas
  #4 clusters chosen because in general they corresponds to vertical and almost vertical lines in symbols
  n_cl=min(4, len(thetas))

  thetas=np.float32(thetas)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
  flags = cv2.KMEANS_PP_CENTERS
  compactness,idx,centers = cv2.kmeans(thetas,n_cl,criteria,10,flags)
  idx=idx.flatten()

  data=[list(thetas[idx==i]) for i in xrange(n_cl)]
  data=sorted([d for d in data if d]) #sort before merge

  changed=True
  while changed:
    changed=False
    for i in xrange(0, len(data)-1):
      #merge if less than 5 degree away
      if abs(np.median(data[i])-np.median(data[i+1]))<0.17453/2:
        data[i]+=data[i+1]
        data.pop(i+1)
        changed=True
        break

  return data

def vertical_angles(img):
  ths=thresholded(img)

  thetas1=vertical_angles_from_contours(ths[-1])
  thetas2=vertical_angles_from_edges(img)

  thetas1_=[np.median(c) for c in thetas1]
  thetas2_=[np.median(c) for c in thetas2]

  if len(thetas1_)==0:
    return sorted(thetas2_, reverse=False, key=lambda x: np.abs(x))

  #get consistent angles
  th=[]
  for theta2 in thetas2_:
    if np.min(np.abs(theta2-thetas1_))<0.1:
      th+=[theta2]

  #commented for test
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
