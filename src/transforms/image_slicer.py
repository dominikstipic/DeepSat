from PIL import Image

import torchvision

####### CROP STRATEGY #################

def _covered_area(dim, kernel_size, stride):
    t = int((dim-kernel_size)/stride + 1)
    x_t = kernel_size + stride*(t-1)
    return x_t

def crop(img, kernel_size, stride):
    w,h = img.size
    covered_width, covered_height = _covered_area(w, kernel_size, stride), _covered_area(h, kernel_size, stride),
    t = torchvision.transforms.CenterCrop([covered_width, covered_height])
    img = t(img)
    return img


####### PAD STRATEGY #################

def add_margin(pil_img, top, right, bottom, left):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height))
    result.paste(pil_img, (left, top))
    return result

def pad(img, kernel_size, stride):
    w,h = img.size
    covered_width, covered_height = _covered_area(w, kernel_size, stride), _covered_area(h, kernel_size, stride)
    covered_width_next, covered_height_next = covered_height + stride, covered_width + stride 
    gap_width, gap_height = covered_width_next - w, covered_height_next - h
    if gap_height == 0 and gap_width == 0: return img
    left = int(gap_width/2)
    right = gap_width - left
    top = int(gap_height/2)
    bottom = gap_height - top
    img = add_margin(img, top, right, bottom, left)
    return img   

############################################3

__strategy__ = {
                "crop" : crop, 
                "pad" :  pad
                }

class KernelSlicer(object):
  def __init__(self, kernel_size, overlap_perc, strategy="crop"):
    assert overlap_perc >= 0 and overlap_perc < 1, "An overlap percentage must be between 0 and 1"
    assert strategy in __strategy__.keys(), "strategy must be crop or pad!"
    self.strategy = __strategy__[strategy]
    self.kernel_size = kernel_size
    overlap_size = int(overlap_perc * kernel_size)
    self.stride = kernel_size - overlap_size

  def bbox(self,img):
    img = self.strategy(img, self.kernel_size, self.stride)
    w,h = img.size
    crops = []
    for y1 in range(0, h - self.kernel_size + 1, self.stride):
      for x1 in range(0, w - self.kernel_size + 1, self.stride):
        x2, y2 = x1 + self.kernel_size - 1, y1 + self.kernel_size - 1
        box = (x1,y1,x2+1,y2+1)
        crop = img.crop(box)
        crops.append(crop)
    return crops

  def __call__(self, xs):
    x,y = xs
    x_crops = self.bbox(x)
    y_crops = self.bbox(y)
    return x_crops, y_crops


