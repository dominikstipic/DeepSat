import torch.nn as nn
import torch.nn.functional as F
import warnings
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as cp
from math import log2
from itertools import chain

def upsample_bilinear(img, size):
    return F.interpolate(img, size, mode='bilinear', align_corners=False)

def upsample_nearest(fixed_size):
    def inner(img):
        return  F.interpolate(img, mode='nearest', size=fixed_size)
    return inner


batchnorm_momentum = 0.01 / 2

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1, drop_rate=.0):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in , num_maps_out, kernel_size=k, 
                                          padding=padding, bias=bias, dilation=dilation))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))

class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, fixed_size=None, k=3):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=k, batch_norm=use_bn)
        self.use_skip = use_skip
        self.upsampling_method = upsample_bilinear
        if fixed_size is not None:
            self.upsampling_method = upsample_nearest(fixed_size)
            warnings.warn("Fixed upsample size", UserWarning)

    def forward(self, x, skip):
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x

def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet18-5c106cde.pth"), strict=False)
    return model

def convkxk(in_planes, out_planes, stride=1, k=3):
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride, padding=k // 2, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = norm(conv(x))
        if relu is not None:
            x = relu(x)
        return x
    return bn_function


def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, bn_class=nn.BatchNorm2d, levels=3):
        super(BasicBlock, self).__init__()
        self.conv1 = convkxk(inplanes, planes, stride)
        self.bn1 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.relu_inp = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convkxk(planes, planes)
        self.bn2 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.num_levels = levels

    def forward(self, x, level):
        residual = x
        bn_1 = _bn_function_factory(self.conv1, self.bn1[level], self.relu_inp)
        bn_2 = _bn_function_factory(self.conv2, self.bn2[level])

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        relu = self.relu(out)
        return relu, out

class ResNet(nn.Module):
    def _make_layer(self, block, planes, blocks, stride=1, bn_class=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       bn_class(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.efficient, bn_class=bn_class,
                            levels=self.pyramid_levels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_class=bn_class, levels=self.pyramid_levels, efficient=self.efficient))
        return nn.Sequential(*layers)

    def __init__(self, block, layers, *, num_features=128, pyramid_levels=3, use_bn=True, k_bneck=1, k_upsample=3,
                 efficient=False, upsample_skip=True, scale=1, detach_upsample_skips=(),
                 pyramid_subsample='bicubic', target_size=None, output_stride=4, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.pyramid_levels = pyramid_levels
        self.num_features = num_features
        self.pyramid_subsample = pyramid_subsample
        self.target_size = target_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn_class = nn.BatchNorm2d 
        self.bn1 = nn.ModuleList([bn_class(64) for _ in range(pyramid_levels)])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        bottlenecks = []
        self.layer1 = self._make_layer(block, 64, layers[0], bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]

        num_bn_remove = max(0, int(log2(output_stride) - 2))
        self.num_skip_levels = self.pyramid_levels + 3 - num_bn_remove 
        bottlenecks = bottlenecks[num_bn_remove:]

        self.upsample_bottlenecks = nn.ModuleList(bottlenecks[::-1])
        num_pyr_modules = 2 + pyramid_levels - num_bn_remove
        target_sizes = [None] * num_pyr_modules
        self.upsample_blends = nn.ModuleList([_UpsampleBlend(num_features, use_bn=use_bn, use_skip=upsample_skip,
                                                             fixed_size=ts, k=k_upsample) for ts in target_sizes])
        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.bn1]
        self.random_init = [self.upsample_bottlenecks, self.upsample_blends]
        self._init_weights()
        
    def _init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def random_init_params(self):
      return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
      return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers, idx):
        skip = None
        for l in layers:
            x,skip = l(x, idx)
        return x, skip

    def forward_down(self, image, skips, level):
        x = self.conv1(image)
        x = self.bn1[level](x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for k, layer in enumerate(layers):
          x, skip = self.forward_resblock(x, layer, level)
          features += [skip]
      
        skip_feats = [b(f) for b, f in zip(self.upsample_bottlenecks, reversed(features))]
        for i, s in enumerate(reversed(skip_feats)):
          skips[level + i] += [s]
        return skips

    def forward(self, image):
        pyramid = [image]
        for l in range(1, self.pyramid_levels):
          resized_img = F.interpolate(image, scale_factor=1 / 2 ** l, mode=self.pyramid_subsample, align_corners=False, recompute_scale_factor=False)
          pyramid += [resized_img]

        skips = [[] for _ in range(self.num_skip_levels)]
        for level, img in enumerate(pyramid):
            skips = self.forward_down(img, skips, level)
        
        skips = skips[::-1]
        x = skips[0][0]
        for i, (skip, blend) in enumerate(zip(skips[1:], self.upsample_blends)):
            x = blend(x, sum(skip))
        return x,pyramid




