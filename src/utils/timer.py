import time
import gc

import torch

class CodeTimer(object):
  def __init__(self, active):
    self.active = active
    self.time = None
    self._start_time = 0
    self._end_time = 0
  
  @staticmethod
  def _start_timer():
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.synchronize()
    return time.perf_counter()
  
  @staticmethod
  def _end_timer():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.perf_counter()

  def __enter__(self):
    if self.active: self._start_time = self._start_timer()
    return self
  
  def __exit__(self, *args):
    if self.active: 
      self._end_time = self._end_timer()
      self.time = self._end_time - self._start_time