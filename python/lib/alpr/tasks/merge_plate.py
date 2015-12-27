# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np
from alpr.utils import overlap

class TaskMergePlate(Task):
  def __init__(self, localization):
    self.localization=localization

  def execute(self):
    loc=self.localization[0]
    plate_text=""

    changed=True
    ovlp=[[(overlap(li[1], lj[1])+0.0)/(li[1][2]*li[1][3]) for lj in loc] for li in loc]
    while changed:
      changed=False
      for j in xrange(len(loc)):
        if loc[j] is None:
          continue
        for i in xrange(len(loc)):
          if loc[i] is None:
            continue

          if i==j:
            continue

          if loc[i][0]==loc[j][0] and (ovlp[i][j]>0.75 or ovlp[j][i]>0.75):
            #drop symbols detected on multiple scales
            loc[i]=None
            changed=True
            continue

        if changed:
          break

    #prune different intersecting symbols, use score regularized with area as selector
    #make no sence theoreticaly but works
    #FIXME probably should use Platt scaling or isotonic regression
    changed=True
    while changed:
      changed=False
      for j in xrange(len(loc)):
        if loc[j] is None:
          continue
        for i in xrange(len(loc)):
          if loc[i] is None:
            continue

          if i==j:
            continue
          if ovlp[i][j]>0.839:
            if loc[i][2]*loc[i][1][2]*loc[i][1][3]> loc[j][2]*loc[j][1][2]*loc[j][1][3]:
              loc[j]=None
            else:
              loc[i]=None
            changed=True
            continue

    loc=[l for l in loc if l is not None]

    o=[[(overlap(li[1], lj[1])+0.0)/(li[1][2]*li[1][3]) for lj in loc] for li in loc]

    for i in xrange(len(o)):
      for j in xrange(len(o)):
        if i!=j:
          if o[i][j]>0.16:
            #print i, j ,loc[i], loc[j], o[i][j]#, (overlap(loc[i], loc[j], True)+0.0)#/(li[1][2]*li[1][3])
            pass

    #FIXME works only for type1 russian plate
    if len(loc)>=8:
      #by position
      for i in xrange(len(loc)):
        letter, box, score = loc[i]
        X,Y,W,H = box
        if letter=='0O':
          if i==0:
            letter='O'
          if i>=1 and i<=3:
            letter='0'
          if i>=4 and i<=5:
            letter='O'
          if i>=6:
            letter='0'

        loc[i]=(letter, (X,Y,W,H), score)
    elif len(loc)==0:
      pass
    else:
      #by height statistics
      hs=np.float32(np.array([p[1][3] for p in loc]))
      n_cl=min(2, len(loc))
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
      flags = cv2.KMEANS_PP_CENTERS
      compactness,idx,centers = cv2.kmeans(hs,n_cl,criteria,10,flags)
      idx=idx.flatten()
      n_cl=len(list(set(idx)))

      data=[list(hs[idx==i]) for i in xrange(n_cl)]
      #bigger should have index 1
      if n_cl>1 and np.mean(data[0])> np.mean(data[1]):
        idx=1-idx
      for i in xrange(len(loc)):
        letter, box, score = loc[i]
        X,Y,W,H = box
        if letter=='0O':
          if i<=len(loc)-3:
            if idx[i]==1:
              letter='0'
            else:
              letter='O'
          else:
            letter='0'
        loc[i]=(letter, (X,Y,W,H), score)


    return TaskResultMergePlate(''.join([l[0] for l in loc if l is not None]))

class TaskResultMergePlate(TaskResult):
  def __init__(self, plate):
    self.plate=plate
