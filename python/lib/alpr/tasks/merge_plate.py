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
    o=[]
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

          if loc[i][0]==loc[j][0] and (ovlp[i][j]>0.7 or ovlp[j][i]>0.7) and loc[i][0] in ['8', '0O', 'C', 'M', '7', '1']:
            #drop symbols detected on multiple scales
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['HM', 'H'] and loc[j][0]=='M' and (ovlp[i][j]>0.85 or ovlp[j][i]>0.85):
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['HM'] and loc[j][0]=='H' and (ovlp[i][j]>0.85 or ovlp[j][i]>0.85):
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['C'] and loc[j][0] in ['8B', '8', 'B'] and ovlp[i][j]>0.9:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['C'] and loc[j][0] in ['9'] and ovlp[i][j]>0.75:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['7'] and loc[j][0] in ['1'] and ovlp[i][j]>0.9:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['A'] and loc[j][0] in ['8'] and ovlp[i][j]>0.85:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['E'] and loc[j][0] in ['B'] and (ovlp[i][j]>0.9 or ovlp[j][i]>0.9):
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['0O'] and loc[j][0] in ['6', '9', '8'] and ovlp[i][j]>0.9:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['1'] and loc[j][0] in ['4'] and ovlp[i][j]>0.75:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['8B', '8'] and loc[j][0]=='B' and (ovlp[i][j]>0.85 or ovlp[j][i]>0.85):
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['8B'] and loc[j][0]=='8' and (ovlp[i][j]>0.85 or ovlp[j][i]>0.85):
            loc[i]=None
            changed=True
            continue

          #FIXME retrain Y
          if loc[i][0] in ['Y'] and loc[j][0]=='X' and ovlp[i][j]>0.9:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['1'] and loc[j][0]=='M' and ovlp[i][j]>0.6:
            loc[i]=None
            changed=True
            continue

          if loc[i][0] in ['C'] and loc[j][0]=='9' and ovlp[i][j]>0.6:
            loc[i]=None
            changed=True
            continue

        o=ovlp

        if changed:
          changed=True
          break

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
