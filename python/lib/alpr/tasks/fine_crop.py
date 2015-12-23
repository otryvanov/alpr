# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
from alpr.tasks.zone_transform import *
import cv2
import numpy as np

class TaskFineCrop(Task):
  def __init__(self, img, left_border, right_border, debug=None):
    self.img=img
    self.debug=debug
    self.left_border=left_border
    self.right_border=right_border
    if debug is None:
      def debug(img, name):
        pass
      self.debug=debug

  def execute(self):
    #detect vertical boundary using only central horizontal part to avoid mess at borders
    v_splits=self.detect_vertical_boundary(self.img[:,int(self.img.shape[1]*0.15):int(self.img.shape[1]*0.85)])

    results=[]
    for v_s in v_splits[:1]: #FIXME move this to upper level
      v_bounded=self.img[v_s[1]-1:v_s[2]+1,:]
      #FIXME test redetect
      #v_splits2=self.detect_vertical_boundary(v_bounded)
      #v_bounded=v_bounded[v_splits2[0][1]-1:v_splits2[0][2]+1,:]
      self.debug(v_bounded, "af_vb")

      h_splits=self.detect_horizontal_boundary(v_bounded)
      for h_s in h_splits:
        h_bounded=v_bounded[:,h_s[1]-1:h_s[2]+1]
        h_bounded=cv2.resize(h_bounded, (520/4, 112/4)) #FIXME possible multiply by 1.2
        self.debug(h_bounded, "af_hb")

        results+=[TaskResultFineCrop(h_bounded[2:-2,2:-2])]

    if len(results)==0:
      results+=[TaskResultFineCrop(self.img[1:-1,1:-1])]

    return results

  def detect_vertical_boundary(self, img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,2)
    th=np.float32(th)
    img_=th.copy()

    #remove small defects
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    img_ = cv2.erode(img_, h_kernel, iterations = 1)

    #destroy content except long horizontal lines
    #FIXME possible erode up to 10 times to retain plate subline
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_.shape[1]*0.25),1)) #magical values
    img_ = cv2.dilate(img_, h_kernel, iterations = 1)
    img_ = cv2.erode(img_, h_kernel, iterations = 1)

    hl=np.mean(255-img_, 1)
    if max(hl)==0:
      #FIXME we probably hit opencv bug
      return [(gray.shape[0]-2, 1, gray.shape[1]-1)]
    hl=hl/np.max(hl)*255
    hl=hl<np.mean(hl)+np.std(hl)
    #FIXME this makes more sense than previous line but does not work
    #hl=hl<127

    limit_h=0.4
    min_y=[n for n in xrange(2,len(hl)-2) if hl[n] and hl[n+1] and not hl[n-1]]
    max_y=[n for n in xrange(len(hl)-3, 2,-1) if hl[n] and hl[n-1] and not hl[n+1]]
    min_y=[n for n in min_y if n<img.shape[0]*limit_h]
    max_y=[n for n in max_y if n>img.shape[0]*(1-limit_h)]

    min_height=14
    splits=sorted([(j-i, i, j) for i in min_y+[1] for j in max_y+[len(hl)-2] if (j-i>min_height)])

    self.debug(th, "af_vb_th")
    self.debug(img_, "af_vb_im")

    if len(splits)==0:
      return [(gray.shape[0]-2, 1, gray.shape[1]-1)]

    return splits

  def detect_horizontal_boundary(self, img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,2)
    th=np.float32(th)
    img_=th.copy()

    #destroy long vertical lines in central part
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
    h1=int(img.shape[1]*0.25)
    h2=int(img.shape[1]*0.75)
    img_[:,h1:h2]=cv2.dilate(img_[:,h1:h2], h_kernel, iterations = 1)

    expected_length=img.shape[0]/112.0*520.0 #520 x 112 is standard plate geometry

    #destroy content except for long vertical lines
    #magical values, for some reasons higher values sometimes fails (opencv bug?)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(img.shape[0]*0.75)))
    img_ = cv2.dilate(img_, v_kernel, iterations = 1)
    img_ = cv2.erode(img_, v_kernel, iterations = 1)

    vl=np.mean(255-img_, 0)
    if max(vl)==0:
      #FIXME we probably hit opencv bug
      return [(gray.shape[1]-3, 1, gray.shape[1]-2)]
    vl=vl/np.max(vl)*255
    vl=vl<np.mean(vl)+np.std(vl)

    limit_v=0.3
    min_x=[n for n in xrange(2,len(vl)-2) if vl[n] and vl[n+1] and not vl[n-1]]
    max_x=[n for n in xrange(len(vl)-3, 2,-1) if vl[n] and vl[n-1] and not vl[n+1]]
    min_x=[n for n in min_x if n<img.shape[1]*limit_v and n> img.shape[1]*0.1]
    max_x=[n for n in max_x if n>img.shape[1]*(1-limit_v) and n< img.shape[1]*0.9]

    min_length=expected_length*0.5 #magic number, volatile, should move to params
    max_length=expected_length*1.5

    #fallback
    if len(min_x)==0:
      min_x+=[1]
    if len(max_x)==0:
      max_x+=[len(vl)-2]

    splits=[(j-i, i, j) for i in min_x for j in max_x if (j-i>min_length) and (j-i< max_length)]

    min_x=list(set([s[1] for s in splits]))
    max_x=list(set([s[2] for s in splits]))

    winSize = (20, 30)
    blockSize = (4,6)
    blockStride = (2,3)
    cellSize = (2,3)
    nbins=9

    winStride = (20,30)
    padding = (0,0)

    gray_padding=int(gray.shape[0]/3.0)
    gray_=cv2.copyMakeBorder(gray, 0, 0, gray_padding, gray_padding, cv2.BORDER_REPLICATE)

    min_x_score={}

    for i in min_x:
      box=(int(i+gray_padding-gray.shape[0]/6.0),0,int(gray.shape[0]/3.0),gray.shape[0])
      sim=gray_[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
      sim=cv2.resize(sim, winSize, interpolation = cv2.INTER_CUBIC)

      hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
      desc = hog.compute(sim, winStride, padding, ((0, 0),))

      score=self.left_border.predict(desc, returnDFVal=True)
      min_x_score[i]=(score, sim)

    max_x_score={}
    for j in max_x:
      box=(int(j+gray_padding-gray.shape[0]/6.0),0,int(gray.shape[0]/3.0)+3,gray.shape[0])
      sim=gray_[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
      sim=cv2.resize(sim, winSize, interpolation = cv2.INTER_CUBIC)

      hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
      desc = hog.compute(sim, winStride, padding, ((0, 0),))

      score=self.right_border.predict(desc, returnDFVal=True)
      max_x_score[j]=(score, sim)

    splits=sorted(splits, key=lambda x: min_x_score[x[1]][0]+max_x_score[x[2]][0], reverse=True)
    splits=splits[:1]

    if len(splits)>0:
      spt, i, j= splits[0]

      if min_x_score[i][0]>0:
        self.debug(min_x_score[i][1], "left_b_good_"+str(i))
      else:
        self.debug(min_x_score[i][1], "left_b_bad_"+str(i))

      if max_x_score[j][0]>0:
        self.debug(max_x_score[j][1], "right_b_good_"+str(j))
      else:
        self.debug(max_x_score[j][1], "right_b_bad_"+str(j))

    self.debug(th, "af_hb_th")
    self.debug(img_, "af_hb_im")

    #FIXME should return something
    return splits

class TaskResultFineCrop(TaskResult):
  def __init__(self, img):
    self.img=img
