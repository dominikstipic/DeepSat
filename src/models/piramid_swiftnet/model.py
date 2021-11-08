import itertools

from src.sat_model import Sat_Model
from . import piramid_resnet

class PiramidSwiftnet(Sat_Model):
  def __init__(self, num_classes):
    super(PiramidSwiftnet, self).__init__()
    self.num_classes = num_classes
    self.backbone = piramid_resnet.resnet18(pretrained=True,
                                            pyramid_levels=3,
                                            k_upsample=3,
                                            scale=1,
                                            k_bneck=1,
                                            output_stride=4,
                                            efficient=True)
    self.logits = piramid_resnet._BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=True, k=1, bias=True)   

  def random_init_params(self):
    params = [self.logits.parameters(), self.backbone.random_init_params()]
    return itertools.chain(*(params))

  def fine_tune_params(self):
    return self.backbone.fine_tune_params()

  def copy(self):
    other = self.__class__(self.num_classes)
    return other
    
  def forward(self, image):
    image_size = image.shape[-2:]
    features,_ = self.backbone(image)
    logits = self.logits.forward(features)
    logits = piramid_resnet.upsample_bilinear(logits, image_size)
    return logits