# -*- encoding: utf-8 -*-

from alpr.decorators import memoize
import itertools

#calculates area overlap of two boxes (format (x,y,w,h)
def overlap(box1, box2):
  #box consist of x,y,w,h
  x_overlap = max(0, min(box1[0]+box1[2],box2[0]+box2[2]) - max(box1[0],box2[0]))
  y_overlap = max(0, min(box1[1]+box1[3],box2[1]+box2[3]) - max(box1[1],box2[1]))
  return x_overlap * y_overlap

#difference of two intervals
def range_diff(range1, range2):
  start1, size1 = range1
  start2, size2 = range2
  endpoints = sorted((start1, start2, start1+size1, start2+size2))
  result = []
  if endpoints[0] == start1:
    result.append((endpoints[0], endpoints[1]-endpoints[0]))
  if endpoints[3] == start1+size1:
    result.append((endpoints[2], endpoints[3]-endpoints[2]))
  return result

#difference of two interval sets
def range_diff_many(ranges1, ranges2):
  for r2 in ranges2:
    ranges1 = list(itertools.chain(*[range_diff(r1, r2) for r1 in ranges1]))
  return ranges1
