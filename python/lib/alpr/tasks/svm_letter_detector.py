# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np
import re
import time

class TaskSVMLetterDetector(Task):
  def __init__(self, img, hog_descriptor, debug=None):
    self.img=img
    self.hog_descriptor=hog_descriptor
    self.debug=debug
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

  def execute(self):
    img=self.img
    gray=cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,2)

    #guess white color
    white_c=np.median(gray)
    white_c=np.median(gray[gray>white_c])

    #extend borders
    img=cv2.copyMakeBorder(img, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=(int(white_c), int(white_c), int(white_c)))
    gray=cv2.copyMakeBorder(gray, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=int(white_c))
    th = cv2.copyMakeBorder(th, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=255)

    #FIXME destroy russian flag
    kernel=np.ones((3,3),np.uint8)
    img_=th[20:, 100:]
    img_ = cv2.dilate(img_, kernel, iterations = 1)
    img_ = cv2.erode(img_, kernel, iterations = 1)
    th[20:, 100:]=img_

    plate=[]

    contours, _ = cv2.findContours(th.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key= lambda x: cv2.contourArea(x), reverse=False)

    m=0
    plate_area=img.shape[0]*img.shape[1]
    for cnt in contours:
      m+=1
      X,Y,W,H = cv2.boundingRect(cnt)
      cnt_box_area=W*H

      #select reasonable bounding boxes
      if cnt_box_area>plate_area/8.0 or cnt_box_area<70 or (W+0.0)/H>2.0:
        continue

      gray_=gray.copy()
      cv2.drawContours(gray_,[cnt],0,(0,127,127),2)
      self.debug(gray_, "svm_cnt_"+str(m))

      #loose contour bounding box
      r_x=0.7 if (H+0.0)/W>1.6 else 0.5
      r_y=0.8 if (H+0.0)/W<1.3 and (H+0.0)/W>0.9 else 0.5
      X,Y,W,H=fit_box(X,Y,W,H, r_x, r_y, gray)

      lp=gray[Y:Y+H,X:X+W]
      self.debug(lp, "svm_crp_"+str(m))

      sss=3
      if (X+W/2.0)>img.shape[1]*0.5:
        sss=3
      if (X+W/2.0)>img.shape[1]*0.75:
        sss=4
      fscale=30.0*sss/img.shape[0]
      lp=cv2.resize(lp, (0,0), fx=fscale, fy=fscale)
      Yf=int(Y*fscale)
      Xf=int(X*fscale)
      Wf=int(W*fscale)
      Hf=int(H*fscale)

      if Wf<20 or Hf<30:
        continue

      cnt_letters=0
      for letter in ['9','7','1','0O','8','8B','HM','2','3','4','5','6','A','B','C','E','H','K','M','P','T','X','Y']:
        found, F=self.hog_descriptor[letter].detectMultiScale(lp, 1.05)
        n=0

        early_break=False
        for j in xrange(len(found)):
          if letter in ['A','B','C','E','H','K','M','P','T','X','Y']:#,'8', '8B']:
            cnt_letters+=1
          (x,y,w,h)=found[j]
          if (w*h+0.0)/(Wf*Hf)>0.33:
            early_break=True
          f=F[j][0]

          lpl=lp[y:y+h,x:x+w]
          lname="svm_letter_"+letter+"_"+str(n)+"_"+str(int(time.time()))+".jpg"
          self.debug(lpl, "svm_let_"+letter+"_"+str(n))
          n+=1
          Xn=X+x/fscale
          Yn=Y+y/fscale
          Wn=w/fscale
          Hn=h/fscale
          plate+=[(Xn+Wn/2.0, letter, (Xn,Yn,Wn,Hn), f)]
        if len(found)>=1 and letter not in ['H', 'HM', 'M', '8B', 'B', '8', 'E', 'C', '1', '0O', 'P'] and early_break:
          break

      #FIXME remove this mess
      if cnt_letters==0:
        lp=cv2.resize(lp, (0,0), fx=1.0, fy=1.2) # if plate is contain subline

        for letter in ['9','7','1','0O','8','8B','HM','2','3','4','5','6','A','B','C','E','H','K','M','P','T','X','Y']:
          found, F=self.hog_descriptor[letter].detectMultiScale(lp, 1.05)
          n=0

          early_break=False
          for j in xrange(len(found)):
            (x,y,w,h)=found[j]
            if (w*h+0.0)/(Wf*Hf)>0.33:
              early_break=True
            f=F[j][0]

            lpl=lp[y+1:y+1+h,x:x+w]
            lname="svm_letter_s_"+letter+"_"+str(n)+"_"+str(int(time.time()))+".jpg"
            n+=1
            Xn=X+x/fscale
            Yn=Y+y/fscale/1.2
            Wn=w/fscale
            Hn=h/fscale/1.2
            plate+=[(Xn+Wn/2.0, letter, (Xn,Yn,Wn,Hn), f)]
          if len(found)>=1 and letter not in ['H', 'HM', 'M', '8B', 'B', '8', 'E', 'C', '1', '0O', 'P'] and early_break:
            break

    return TaskResultSVMLetterDetector([l[1] for l in sorted(plate)], [l[1:] for l in sorted(plate)])

class TaskResultSVMLetterDetector(TaskResult):
  def __init__(self, plate, localization):
    self.plate=plate
    self.localization=localization


def fit_box(X,Y,W,H, scale_x, scale_y, img):
  #upscale box and fit it back into img

  Xmin=int(X+W/2.0-W*(1+scale_x)/2.0)
  Xmax=int(X+W/2.0+W*(1+scale_x)/2.0)
  Ymin=int(Y+H/2.0-H*(1+scale_y)/2.0)
  Ymax=int(Y+H/2.0+H*(1+scale_y)/2.0)
  if Xmin<0:
    Xmin=0
  if Xmax>img.shape[1]:
    Xmax=img.shape[1]
  if Ymin<0:
    Ymin=0
  if Ymax>img.shape[0]:
    Ymax=img.shape[0]

  X=Xmin
  Y=Ymin
  W=Xmax-Xmin
  H=Ymax-Ymin

  return (X,Y,W,H)
