from PIL import Image
import random

import numpy as np
import torch
import torchvision.transforms as torch_transf
import cv2

#######################

class Compose(object):
  def __init__(self, transforms: list):
    self.transforms = transforms
      
  def __call__(self, xs: list) -> list:
    for t in self.transforms:
      xs = t(xs)
    return xs

  def append(self, other_compose): 
    self.transforms += other_compose.transforms

  @staticmethod
  def from_composits(*composits):
    result = Compose([])
    for c in composits:
      result.append(c)
    return result

#######################

class Resize(object):
  def __init__(self, size, mehtods=[Image.BICUBIC, Image.NEAREST]):
    size = [size]*2
    self.img_resize  = torch_transf.Resize(size)
    self.mask_resize = torch_transf.Resize(size, Image.NEAREST)
  
  def __call__(self, xs):
    x,y = xs
    x,y = self.img_resize(x), self.mask_resize(y)
    return x,y

#######################

class Downsample(object):
  def __init__(self, ratio):
    self.ratio = ratio
  
  def __call__(self, xs):
    x,y = xs
    H,W = x.size
    H_new, W_new = int(H*self.ratio), int(W*self.ratio)
    resize = torch_transf.Resize([W_new, H_new])
    x,y = resize(x), resize(y)
    return x,y

#######################       
 
class Cropper(object):
  def __init__(self, size, five_crop=True):
    if five_crop:
        self.transform = torch_transf.FiveCrop(size)
    else:
        self.transform = torch_transf.TenCrop(size)
  
  def __call__(self, xs):
    x,y = xs
    x,y = self.transform(x), self.transform(y)
    return x,y

#######################

class ListShuffler(object):
  def __call__(self, batches):
    batches = list(zip(*batches))
    random.shuffle(batches)
    batches = list(zip(*batches))
    return batches

#######################

class To_Tensor(object):
  def __init__(self, mean: list, std: list, input_type: np.dtype, label_type: np.dtype):
    self.mean = mean
    self.std  = std
    self.input_type = input_type
    self.label_type = label_type

  def _trans(self, img, dtype):
    img = np.array(img, dtype=dtype)
    if len(img.shape) == 3:
      img = np.transpose(img, (2,0,1))
      img = np.ascontiguousarray(img)
      img = torch.from_numpy(img)
      t = torch_transf.Normalize(mean=self.mean, std=self.std)
      img = t(img)
      return img
    else:
      return torch.from_numpy(img)

  def __call__(self, example):
    image, labels = example
    image  = self._trans(image, self.input_type)
    labels = self._trans(labels, self.label_type)
    return image, labels

#######################

class Dilation(object):
  def __init__(self, iterations, kernel_size):
    self.iterations = iterations
    self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

  def dilate(self, img):
    img = np.array(img).astype(np.uint8)
    out = cv2.dilate(img.copy(), self.kernel, iterations=self.iterations+1)
    out = torch.from_numpy(out)
    return out

  def __call__(self, logits):
    if len(logits.shape) == 4:
      predictions = logits.argmax(1)
      batch_size,_,_ = predictions.shape
      for k in range(batch_size):
        predictions[k] = self.dilate(predictions[k])
      return predictions
    else:
      predictions = logits.argmax(0)
      return self.dilate(logits)