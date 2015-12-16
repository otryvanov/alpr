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

    self.search_confusion_map={
      '0O': ['6', '9'],
      '1': ['M', 'K'],
      '2': ['4'],
      '3': [],
      '4': ['2'],
      '5': [],
      '6': [],
      '7': [],
      '8': ['B', 'A'],
      '8B': ['8', 'B'],
      '9': [],
      'A': ['8'],
      'B': [],
      'C': ['E', '8', '8B', 'B'],
      'E': [],
      'H': ['M'],
      'K': ['1'],
      'M': ['H', '1'],
      'P': [],
      'T': [],
      'X': [],
      'Y': [],
      'HM': ['M', 'H']
    }

    self.search_order=['7','9','1','8','0O','8B','HM','2','3','4','5','6','A','B','C','E','H','K','M','P','T','X','Y']
    self.search_yscales=[1.0, 1.2]

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

    letters=['1','2','3','4','5','6','7','8','9','0O','A','B','C','E','H','K','M','P','T','X','Y']
    @memoize_simple
    def max_score(box):
        x,y,w,h=box
        if w*h<min_area or w<min_width or h<min_height:
          return (None, 1.0)

        l_s=[(l, self.test_letter(box, gray, l)) for l in letters]
        return min(l_s, key=lambda x: x[1])

    letter_ligatures=['8dot', 'dotO', 'dotM', 'dotB', 'dotC', 'dotH', 'dotE', 'dotP']
    @memoize_simple
    def max_score_ligatures(box):
        x,y,w,h=box
        if w*h<min_area or w<min_width or h<min_height:
          return (None, 1.0)

        l_s=[(l, self.test_letter(box, gray, l)) for l in letter_ligatures]
        return min(l_s, key=lambda x: x[1])

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

    #are we done?
    #RUSSIAN PLATE TYPE1 SPECIFIC
    #alphas, nums, alphanums=get_stats_symbols(plate)
    #if alphanums>=9:
    #  return TaskResultSVMLetterDetector(plate)

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

    #prune plate, distructive to original
    plate=prune_plate(plate, threshold=0.799) #distructive

    m=200
    for box in get_positional_boxes(gray, plate):
      m+=1
      X,Y,W,H, deep = box
      if W<min_width:
        continue
      b_img=gray[Y:Y+H,X:X+W]

      fscale=2
      if (X+W/2.0)>img.shape[1]*0.5:
        fscale=2
      if (X+W/2.0)>img.shape[1]*0.75:
        fscale=3
      fscale=30.0*fscale/img.shape[0]
      b_img=cv2.resize(b_img, (0,0), fx=fscale, fy=fscale)
      Yf=int(Y*fscale)
      Xf=int(X*fscale)
      Wf=int(W*fscale)
      Hf=int(H*fscale)

      if Wf<20 or Hf<30:
        continue

      self.debug(b_img, "svm2_"+str(m))

      search_yscales=self.search_yscales
      search_list=copy.copy(self.search_order)
      search_confusion=False

      Found=False

      while len(search_list)>0:
        letter=search_list.pop(0)

        for yscale in search_yscales:
          b_img_=cv2.resize(b_img, (0,0), fx=1.0, fy=yscale) # rescale in case a plate contains subline

          found, F=self.hog_descriptor[letter].detectMultiScale(b_img_, 1.05)
          n=0
          for j in xrange(len(found)):
            Found=True
            x,y,w,h=found[j]
            f=F[j][0]

            b_imgl=b_img_[y:y+h,x:x+w]

            if np.abs(yscale-1.0)<0.001:
              self.debug(b_imgl, "svm2_f_"+letter+"_"+str(m)+"_"+str(n))
            else:
              self.debug(b_imgl, "svm2_fs_"+letter+"_"+str(m)+"_"+str(n))

            n+=1

            Xn=X+x/fscale
            Yn=Y+y/fscale/yscale
            Wn=w/fscale
            Hn=h/fscale/yscale
            plate+=[(letter, (Xn,Yn,Wn,Hn), f)]

          if len(found)>0 and not search_confusion and not deep:
            search_confusion=True
            search_list=copy.copy(self.search_confusion_map[letter])
            continue

        deep=True
        if Found and not search_confusion and not deep:
          if yscale!=search_yscales[0]:
            self.search_yscales=[yscale, search_yscales[0]]
          break

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

  #@memoize_test_letter
  def test_letter(self, box, gray, letter):
    svm=self.svm_letters[letter]

    desc=self.compute_hog(box, gray)
    score_=svm.predict(desc, returnDFVal=True)

    return -score_

  #FIXME this is runtime potential problem
  @memoize_compute_hog
  def compute_hog(self, box, gray):
    X,Y,W,H=box
    gray=gray[Y-1:Y+H+1, X-1:X+W+1]

    winSize = (20, 30)
    blockSize = (4,6)
    blockStride = (2,3)
    cellSize = (2,3)
    nbins=9

    winStride = (20,30)
    padding = (0,0)

    gray=cv2.resize(gray, winSize, interpolation = cv2.INTER_CUBIC)

    hog=cv2.HOGDescriptor(winSize, blockSize,blockStride,cellSize, nbins)
    desc = hog.compute(gray, winStride, padding, ((0, 0),))

    return desc

  def get_positional_boxes(self, th):
    sizes=[(0,25, False),(16,20, False),(27,20, False),(37,20, False), (48,20, False), (59, 20, False), (65,35, True), (85,15, True)]
    boxes=[]
    for X, W, deep in sizes:
      X=int((th.shape[1]-1)*X/100.0)
      W=int((th.shape[1]-1)*W/100.0)
      Y=0
      H=th.shape[0]-1
      boxes+=[(X,Y,W,H, True, None)]

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

def get_positional_boxes(gray, plate):

    return [(r[0],0,r[1]-r[0],gray.shape[0], True) for r in get_free_hranges(gray, plate)]

def get_stats_symbols(plate):
  letters=[l[0] for l in plate]

  alphas=0
  nums=0
  alphanums=len(letters)

  return alphas, nums, alphanums

def prune_plate(plate, threshold=0.999):
    changed=True
    ovlp=[[(overlap(li[1], lj[1])+0.0)/(li[1][2]*li[1][3]) for lj in plate] for li in plate]

    while changed:
      changed=False
      for j in xrange(len(plate)):
        if plate[j] is None:
          continue
        for i in xrange(len(plate)):
          if plate[i] is None:
            continue

          if i==j:
            continue

          if plate[i][0]==plate[j][0] and max(ovlp[i][j], ovlp[j][i])> threshold:
            if plate[i][2]>plate[j][2]:
              #drop symbols detected by multiple boxes
              plate[j]=None
              break
            else:
              plate[i]=None

            changed=True

    idx=[i for i in range(len(ovlp)) if plate[i] is not None]

    #debug
    #for i in idx:
    #  for j in idx:
    #    if i!=j:
    #      if ovlp[i][j]>0.201 and plate[i][1]==plate[j][1]:
    #        print "prune_plate", i, j ,plate[i], plate[j], max(ovlp[i][j], ovlp[j][i])
    #        pass

    return [l for l in plate if l is not None]
