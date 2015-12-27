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

    return TaskResultMergePlate(''.join([l[0] for l in loc if l is not None]))

class TaskResultMergePlate(TaskResult):
  def __init__(self, plate):
    self.plate=plate
