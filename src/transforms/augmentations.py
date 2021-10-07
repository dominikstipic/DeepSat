from PIL import Image

import numpy as np
import torchvision.transforms.functional as TF

class Jitter(object):
  def __init__ (self, rotation=[-5,5], scale=[0.7, 2]):
    self.rotation  = rotation
    self.scale_int = scale
    
  def rot_transform(self, rot, img):
    return TF.affine(img, angle = rot, translate=[0,0], scale = 1, shear = [0, 0])
  
  def scale_transform(self, s, img):
    return TF.affine(img, angle = 0, translate=[0,0], scale = s, shear = [0 ,0])

  def rotate(self, xs):
    x,y    = xs
    l,u    = self.rotation
    degree = self.rand(l,u)
    tx,ty  = self.rot_transform(degree,x), self.rot_transform(degree,y)
    return tx,ty
  
  def scale(self, xs):
    x,y = xs
    l,u = self.scale_int
    s   = self.rand(l,u)
    tx,ty   = self.scale_transform(s,x), self.scale_transform(s,y)
    return tx, ty
  
  def rand(self, lower, upper):
    return np.random.uniform(lower, upper)

  def __call__(self, xs):
    x,y = self.rotate(xs)
    x,y = self.scale([x,y])
    return x,y

#######################

class RandomCropper(object):
  def __init__(self, size):
    self.size = size
    
  def _rand_bbox(self, W, H, target_wh):
    try:
      w = np.random.randint(0, W - target_wh + 1)
      h = np.random.randint(0, H - target_wh + 1)
    except ValueError:
      print(f'Exception in RandomSquareCropAndScale: {target_wh}')
      w = h = 0
    return w, h, w + target_wh, h + target_wh

  def __call__(self, xs):
    x,y  = xs
    W,H  = x.size
    bbox = self._rand_bbox(W, H, self.size)
    x, y = x.crop(bbox), y.crop(bbox)
    return x,y

#######################

class RandomSquareCropAndScale:
  def __init__(self, wh, min=.5, max=2., scale_method=lambda scale, wh, size: int(scale * wh)):
    self.wh = wh
    self.min = min
    self.max = max
    self.scale_method = scale_method

  def _rand_location(self, W, H, target_wh, *args, **kwargs):
    try:
      w = np.random.randint(0, W - target_wh + 1)
      h = np.random.randint(0, H - target_wh + 1)
    except ValueError:
      print(f'Exception in RandomSquareCropAndScale: {target_wh}')
      w = h = 0
    return w, h, w + target_wh, h + target_wh

  def crop_and_scale_img(self, img: Image, crop_box, target_size, pad_size, resample):
      target = Image.new(img.mode, pad_size)
      target.paste(img)
      res = target.crop(crop_box).resize(target_size, resample=resample)
      return res

  def __call__(self, example):
      image, labels = example
      scale = np.random.uniform(self.min, self.max)
      W, H = image.size
      box_size = self.scale_method(scale, self.wh, image.size)
      pad_size = (max(box_size, W), max(box_size, H))
      target_size = (self.wh, self.wh)
      crop_box = self._rand_location(pad_size[0], pad_size[1], box_size)
      image  = self.crop_and_scale_img(image,  crop_box, target_size, pad_size, Image.BICUBIC)
      labels = self.crop_and_scale_img(labels, crop_box, target_size, pad_size, Image.NEAREST)
      return image, labels

#######################

class Flipper(object):
  def __init__ (self, probability):
    self.prob = probability
  
  def flip_coin(self):
    v = np.random.uniform(0,1)
    return v > self.prob

  def flip(self, x):
    return x.transpose(Image.FLIP_LEFT_RIGHT)

  def __call__(self, xs: tuple) -> tuple:
    x,y = xs
    if self.flip_coin() :
      return xs
    else:
      x,y = self.flip(x), self.flip(y)
      return x,y

#######################

class RandomRotation(object):
  def __init__ (self, rotation=[-5,5]):
    self.rotation  = rotation
    
  def rot_transform(self, rot, img):
    return TF.affine(img, angle = rot, translate=[0,0], scale = 1, shear = 0)
  
  def rotate(self, xs):
    x,y    = xs
    l,u    = self.rotation
    degree = self.rand(l,u)
    tx,ty  = self.rot_transform(degree,x), self.rot_transform(degree,y)
    return tx,ty
  
  def rand(self, lower, upper):
    return np.random.uniform(lower, upper)

  def __call__(self, xs):
    x,y = self.rotate(xs)
    return x,y