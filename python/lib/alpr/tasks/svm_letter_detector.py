# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np
import copy
from alpr.decorators import *
from alpr.utils import *

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

  def execute(self):
    img=self.img

    #gray is used for classification and search, gray_c only for search
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,8))
    gray_c = clahe.apply(gray)

    #guess 'white' color
    white=np.median(gray)
    white=np.mean(gray[gray>white])
    white_c=np.median(gray_c)
    white_c=np.mean(gray_c[gray_c>white_c])

    window=21
    #adaptive mean
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window, 2)
    #adaptive gaussian
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window, 2)
    #Otsu's
    _,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #adaptive mean
    th_c2 = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window, 2)
    #adaptive gaussian
    th_c3 = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window, 2)
    #Otsu's
    _,th_c4 = cv2.threshold(gray_c,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #extend borders(probably only needed for detectMultiscale)
    img=cv2.copyMakeBorder(img, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=(int(white), int(white), int(white)))
    gray=cv2.copyMakeBorder(gray, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=int(white))
    gray_c=cv2.copyMakeBorder(gray_c, 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=int(white_c))
    ths=[th2, th3, th4, th_c2, th_c3, th_c4]
    for i in xrange(len(ths)):
      ths[i]=cv2.copyMakeBorder(ths[i], 4, 4, 2, 2, cv2.BORDER_CONSTANT, value=255)
    th2, th3, th4, th_c2, th_c3, th_c4 = ths

    self.debug(img, "svm_img")
    self.debug(gray, "svm_gr")
    self.debug(gray_c, "svm_gr_c")
    self.debug(th2, "svm_th2")
    self.debug(th3, "svm_th3")
    self.debug(th4, "svm_th4")
    self.debug(th_c2, "svm_th_c2")
    self.debug(th_c3, "svm_th_c3")
    self.debug(th_c4, "svm_th_c4")

    plate=[]

    min_height=10
    min_width=5
    min_area=70
    epsilon=0.00001

    @memoize
    def max_score_hsplit(box, n=3):
      x,y,w,h=box
      l,s=max_score(box)

      if s<0.0:
        l_s=[(s, [(l,s,box)])]
      else:
        l_s=[(epsilon, [(l,epsilon,box)])]

      if n>1:
        for w0 in xrange(1, w):
          if w0*h<min_area or w0<min_width or h<min_height:
            l0, s0=(None, epsilon)
          else:
            l0, s0=max_score((x,y,w0,h))

          if (w-w0)*h<min_area or (w-w0)<min_width or h<min_height:
            s1, ls1=(epsilon, [(None, epsilon, (x+w0,y,w-w0,h))])
          else:
            s1, ls1=max_score_hsplit((x+w0,y,w-w0,h),n-1)

          if s0>epsilon:
            s0=epsilon

          score=(s0*w0+s1*(w-w0)+0.0)/w

          l_s+=[(score, [(l0, s0, (x,y,w0,h))]+ls1)]
      return min(l_s)

    #functions defined as closures to avoid passing multiple and/or complex arguments
    #which allows memoize_simple use and autoresets after execute comletion

    def compute_hog(box):
      X,Y,W,H=box
      gray_=gray[Y-1:Y+H+1, X-1:X+W+1] #FIXME should check area bounds

      winSize = (20, 30)
      blockSize = (4,6)
      blockStride = (2,3)
      cellSize = (2,3)
      nbins=9

      winStride = (20,30)
      padding = (0,0)

      gray_=cv2.resize(gray_, winSize, interpolation = cv2.INTER_CUBIC)

      hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
      desc = hog.compute(gray_, winStride, padding, ((0, 0),))

      return desc

    letters=['1','2','3','4','5','6','7','8','9','0O','A','B','C','E','H','K','M','P','T','X','Y']
    @memoize_simple
    def max_score(box):
      x,y,w,h=box

      if w*h<min_area or w<min_width or h<min_height:
        return (None, 1.0)

      desc=compute_hog(box)

      l_s=[(l, -self.svm_letters[l].predict(desc, returnDFVal=True)) for l in letters]

      return min(l_s, key=lambda x: x[1])

    letter_ligatures=['8dot', 'dotO', 'dotM', 'dotB', 'dotC', 'dotH', 'dotE', 'dotP']
    @memoize_simple
    def max_score_ligatures(box):
      x,y,w,h=box
      if w*h<min_area or w<min_width or h<min_height:
        return (None, 1.0)

      desc=compute_hog(box)

      l_s=[(l, -self.svm_letters[l].predict(desc, returnDFVal=True)) for l in letter_ligatures]
      return min(l_s, key=lambda x: x[1])

    h1_candidates=[10,5]
    h2_candidates=[16,22]

    @memoize_simple
    def max_score_vsplit(box):
      x,y,w,h=box
      l_s=[]
      min_score=1.0
      min_letter=None
      min_box=(x,y,w,h)

      for h1 in h1_candidates:
        for h2 in h2_candidates:
          l,s=max_score((x,h1,w,h2))
          s=s*h2/(h+0.0)
          if s<min_score:
            min_score=s
            min_letter=l
            min_box=(x,h1,w,h2)

      return min_letter, min_score, min_box

    def max_score_hsplit3(box):
      x,y,w,h=box

      min_score=1.0
      min_letter=None
      min_box=(x,y,w,h)

      for w1 in xrange(0,min(w-min_width,10)):

        for w2 in xrange(min_width, min(w-w1,16)):
          b_=(x+w1,y,w2,h)

          l,s,b=max_score_vsplit(b_)
          s=s/(w+0)*w2
          if s<min_score:
            min_score=s
            min_letter=l
            min_box=b

      return min_score, min_letter, min_box

    #replacing original compute_hog with memoized version
    #will be restored after ligatures detection
    compute_hog_raw=compute_hog
    compute_hog=memoize_simple(compute_hog)

    boxes=[]
    for th in ths:
      boxes+=self.get_boxes_from_contour(th, gray)
    boxes=list(set(boxes)) #get uniq boxes

    #annotate each box with name for debug, letter, score, cropped image
    boxes=[box+(str(box), None, 1.0, None) for box in boxes]

    #search all boxes for letters
    boxes_left=[]
    while boxes:
      X,Y,W,H,m,min_letter,min_score,b_img = boxes.pop()

      b_img=gray[Y-1:Y+H+1, X-1:X+W+1]
      self.debug(b_img, "svm1_t_"+str(m))

      min_letter, min_score=max_score((X,Y,W,H))

      if min_score<0:
        self.debug(b_img, "svm1_f_"+min_letter+"_"+str(m))
        plate+=[(min_letter, (X,Y,W,H), -min_score)]
      else:
        boxes_left+=[(X,Y,W,H,m,min_letter,min_score, b_img)]

    #prune plate, distructive to origianl
    plate=prune_plate(plate, threshold=0.799)

    #are we done?
    #RUSSIAN PLATE TYPE1 SPECIFIC
    alphas, nums, alphanums=get_stats_symbols(plate)
    if alphanums>=9:
      return TaskResultSVMLetterDetector(plate)

    #prune boxes by content
    hranges=get_free_hranges(gray, plate, 2)
    hranges=range_diff_many([(0,gray.shape[1])], [(r[0], r[1]-r[0]+1) for r in hranges])
    hranges=[r for r in hranges if r[1]>0]

    boxes=boxes_left
    boxes_left=[]
    while boxes:
      X,Y,W,H,m,min_letter,min_score,b_img = boxes.pop()

      fr=range_diff_many([(X,W)], hranges)
      for r in fr:
        X, W=r
        if W<min_width:
          continue
        b_img=gray[Y-1:Y+H+1, X-1:X+W+1]
        min_letter, min_score=max_score((X,Y,W,H))
        m_r=str(m)+"_"+str(r)
        boxes_left+=[(X,Y,W,H,m_r,min_letter,min_score, b_img)]
        self.debug(b_img, "svm1_t2_"+str(m_r))

    #search known 'ligatures'
    boxes=boxes_left
    boxes_left=[]
    while boxes:
      X,Y,W,H,m,min_letter,min_score,b_img = boxes.pop()

      min_letter_new, min_score_new=max_score_ligatures((X,Y,W,H))

      if min_score_new<0:
        min_letter=min_letter_new.replace('dot','').replace('O','0O')
        min_score=min_score_new
        self.debug(b_img, "svm1_fl_"+min_letter+"_"+str(m))
        plate+=[(min_letter, (X,Y,W,H), -min_score)]
      else:
        boxes_left+=[(X,Y,W,H,m,min_letter,min_score,b_img)]

    #prune plate, distructive to origianl
    plate=prune_plate(plate, threshold=0.799)

    #are we done?
    #RUSSIAN PLATE TYPE1 SPECIFIC
    alphas, nums, alphanums=get_stats_symbols(plate)
    if alphanums>=9:
      return TaskResultSVMLetterDetector(plate)

    #search by splitting
    boxes=boxes_left
    boxes_left=[]
    while boxes:
      X,Y,W,H,m,min_letter,min_score,b_img = boxes.pop()

      s, splt=max_score_hsplit((X,Y,W,H), n=3)

      for k in xrange(len(splt)):
        letter_s, score_s, box_s=splt[k]
        if score_s<0:
          b_img_s=gray[box_s[1]-1:box_s[1]+box_s[3]+1, box_s[0]-1:box_s[0]+box_s[2]+1]
          self.debug(b_img_s, "svm1_fspl_"+letter_s+"_"+str(m)+"_"+str(k))
          plate+=[(letter_s, box_s, -score_s)]

      if s>0:
        self.debug(b_img, "svm1_nf_"+str(m))

    #restore original compute_hog
    compute_hog_raw=compute_hog_raw

    #prune plate, distructive to original
    plate=prune_plate(plate, threshold=0.799) #distructive
    #plate=sorted(plate, key=lambda x: x[1][0]+x[1][2]/2.0)

    #'bruteforce' search
    hranges=[(r[0], r[1]-r[0]+1) for r in get_free_hranges(gray, plate, 2)]
    h1_cnds=list(set([l[1][1] for l in plate]))
    h2_cnds=list(set([l[1][3] for l in plate]))

    if len(h1_cnds)>2:
      h1_candidates=h1_cnds
    if len(h2_cnds)>2:
      h2_candidates=h2_cnds

    ws=[l[1][2] for l in plate]
    min_width=max(min_width, int(np.floor(min(ws)*0.75)))
    max_width=20

    for r in hranges:
      x,w=r
      if w<min_width:
        continue
      if x==0:
        x+=1
      if x+w==gray.shape[1]:
        w-=1

      scores=[max_score_hsplit3((x+i, 0, min(w-i,max_width), gray.shape[0])) for i in xrange(0,w-min_width,3)]
      for s in [s for s in scores if s[0]<0.0]:
        min_letter, min_score=max_score(s[2])
        b_img=gray[s[2][1]-1:s[2][1]+s[2][3]+1, s[2][0]-1:s[2][0]+s[2][2]+1]

        self.debug(b_img, "svm1_fbf_"+min_letter+"_"+str(s[2]))
        plate+=[(min_letter, s[2], -min_score)]

    plate=prune_plate(plate, threshold=0.799) #distructive

    return TaskResultSVMLetterDetector(plate)

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
      if (W+0.0)/H>3.0 or (H+0.0)/W>3.0:
        continue
      if cnt_box_area>plate_area/6 or cnt_box_area<70 and prune:
        continue

      boxes+=[(X,Y,W,H)]

    return boxes

class TaskResultSVMLetterDetector(TaskResult):
  def __init__(self, plate):

    self.plate=plate
    self.localization=sorted(plate, key=lambda x: x[1][0]+x[1][2]/2.0)

def get_free_hranges(gray, plate, pad=3):
  mask=np.zeros(gray.shape, np.uint8)

  if len(plate)==0:
    return [(0,gray.shape[1])]

  for p in plate:
    l, box, score=p
    cv2.rectangle(mask, (box[0]+pad, box[1]), (box[0]+box[2]-pad, box[1]+box[3]), 255, thickness=-1)
  h_gaps=np.mean(mask, 0)

  #from matplotlib import pyplot as plt
  #plt.imshow(mask,'gray')
  #plt.show()
  #plt.imshow(gray,'gray')
  #plt.show()

  h_med=np.median(h_gaps[h_gaps>0])
  h_gaps=(h_gaps>h_med/2)+0

  def ranges(markers):
     lm=len(markers)
     h1=np.array([i for i in range(lm) if markers[i]==0 and (i==0 or markers[i-1]==1)])
     h2=np.array([i for i in range(lm) if markers[i]==0 and (i==lm-1 or markers[i+1]==1)])

     return [(h, np.min(h2[h2>=h])+1) for h in h1]

  rgs=[r for r in ranges(h_gaps) if (r[1]-r[0]>5)] #min_width, REFACTOR

  return rgs

def get_stats_symbols(plate):
  letters=[l[0] for l in plate]

  alphas=0
  nums=0
  alphanums=len(letters)

  return alphas, nums, alphanums

def prune_plate(plate, threshold=0.999):
  #plane entries format is (score, box, letter)
  changed=True
  ovlp=[[(overlap(li[1], lj[1])+0.0)/(li[1][2]*li[1][3]) for lj in plate] for li in plate]

  while changed:
    changed=False
    for j in xrange(len(plate)):
      if plate[j] is None:
        continue
      for i in xrange(j+1, len(plate)):
        if plate[i] is None:
          continue

        #drop symbols detected by multiple boxes
        if plate[i][0]==plate[j][0] and max(ovlp[i][j], ovlp[j][i])> threshold:
          changed=True
          if plate[i][2]>plate[j][2]:
            plate[j]=None
            break
          else:
            plate[i]=None

  #debug
  #idx=[i for i in range(len(ovlp)) if plate[i] is not None]
  #for i in idx:
  #  for j in idx:
  #    if i!=j:
  #      if ovlp[i][j]>0.201 and plate[i][1]==plate[j][1]:
  #        print "prune_plate", i, j ,plate[i], plate[j], max(ovlp[i][j], ovlp[j][i])
  #        pass

  return [l for l in plate if l is not None]
