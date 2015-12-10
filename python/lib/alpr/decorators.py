# -*- encoding: utf-8 -*-

import functools

def memoize(obj):
  cache = obj.cache = {}

  @functools.wraps(obj)
  def memoizer(*args, **kwargs):
    key = str(args) + str(kwargs)
    if key not in cache:
      cache[key] = obj(*args, **kwargs)
    return cache[key]
  return memoizer

def memoize_test_letter(obj):
  cache = obj.cache = {}

  @functools.wraps(obj)
  def memoizer(*args, **kwargs):
    key = str(args[1])+args[3]+str(hash(args[2].tostring()))
    if key not in cache:
      cache[key] = obj(*args, **kwargs)
    return cache[key]
  return memoizer

def memoize_compute_hog(obj):
  cache = obj.cache = {}

  @functools.wraps(obj)
  def memoizer(*args, **kwargs):
    key = str(args[1])+str(hash(args[2].tostring()))
    if key not in cache:
      cache[key] = obj(*args, **kwargs)
    return cache[key]
  return memoizer
