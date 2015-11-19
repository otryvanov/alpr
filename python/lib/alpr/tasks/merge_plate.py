# -*- encoding: utf-8 -*-

from alpr.tasks.common import *
import cv2
import numpy as np
import re
import time
import collections

class TaskMergePlate(Task):
  def __init__(self, localization):
    self.localization=localization

  def execute(self):
    loc=self.localization[0]
    plate_text=""

    changed=True
    o=[]
    while changed:
      changed=False
      ovlp=[[(overlap(li, lj)+0.0)/(li[1][2]*li[1][3]) for lj in loc] for li in loc]
      for i in xrange(len(loc)):
        for j in xrange(len(loc)):
          if i==j:
            continue

          if loc[i][0]==loc[j][0] and (ovlp[i][j]>0.85 or ovlp[j][i]>0.85):
            #drop symbols detected on multiple scales
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['HM', 'H'] and loc[j][0]=='M' and (ovlp[i][j]>0.85 or ovlp[j][i]>0.85):
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['HM'] and loc[j][0]=='H' and (ovlp[i][j]>0.85 or ovlp[j][i]>0.85):
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['C'] and loc[j][0] in ['8B', '8', 'B'] and ovlp[i][j]>0.9:
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['7'] and loc[j][0] in ['1'] and ovlp[i][j]>0.9:
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['A'] and loc[j][0] in ['8'] and (ovlp[i][j]>0.9 or ovlp[j][i]>0.9):
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['E'] and loc[j][0] in ['B'] and (ovlp[i][j]>0.9 or ovlp[j][i]>0.9):
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['0O'] and loc[j][0] in ['6', '9', '8'] and ovlp[i][j]>0.9:
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['1'] and loc[j][0] in ['4'] and ovlp[i][j]>0.75:
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['8B', '8'] and loc[j][0]=='B' and (ovlp[i][j]>0.9 or ovlp[j][i]>0.9):
            loc.pop(i)
            changed=True
            break

          if loc[i][0] in ['8B'] and loc[j][0]=='8' and (ovlp[i][j]>0.9 or ovlp[j][i]>0.9):
            loc.pop(i)
            changed=True
            break


          if loc[i][0] in ['1'] and loc[j][0]=='M' and ovlp[i][j]>0.6:
            loc.pop(i)
            changed=True
            break
        o=ovlp

        if changed:
          changed=True
          break

    for i in xrange(len(o)):
      for j in xrange(len(o)):
        if i!=j:
          if o[i][j]>0.16:
            #print i, j ,loc[i][0], loc[j][0], ovlp[i][j]
            pass

    return TaskResultMergePlate(''.join([l[0] for l in loc]))

#    return #TaskResultMergePlate([l[1] for l in sorted(plate)], [l[1:] for l in sorted(plate)])


class TaskResultMergePlate(TaskResult):
  def __init__(self, plate):
    self.plate=plate

def overlap(l1, l2):
#  print l1, l2
  x11=l1[1][0] #left
  y11=l1[1][1] #top
  x12=x11+l1[1][2] #right
  y12=y11+l1[1][3] #bottom

  x21=l2[1][0] #left
  y21=l2[1][1] #top
  x22=x21+l2[1][2] #rigth
  y22=y21+l2[1][3] #bottom

  x_overlap = max(0, min(x12,x22) - max(x11,x21));
  y_overlap = max(0, min(y12,y22) - max(y11,y21));
  overlap = x_overlap * y_overlap;
  return overlap

def merge(l):
  if Set(l)==Set('M', 'H') or Set(l)==Set('M', 'HM') or Set(l)==Set('M', 'HM', 'H'):
    return 'M'
  if Set(l)==Set('H', 'HM'):
    return 'H'
