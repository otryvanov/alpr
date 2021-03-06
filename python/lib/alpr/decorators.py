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

def memoize_simple(f):
  class memodict(dict):
    def __missing__(self, key):
      ret = self[key] = f(key)
      return ret
  return memodict().__getitem__