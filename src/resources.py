# resources.py
# Combined DeepLabV3+ (ResNet backbone) implementation from VainF's repo
# Ready for import in your app.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

# ---------------------------------------------------------------------
# ResNet backbone (from resnet.py)
# ---------------------------------------------------------------------
__all__ = ['ResNet', 'resnet101']

model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# resources.py

# ... (rest of the file remains the same)

class ResNet(nn.Module):
    def __init__(self, block, layers, replace_stride_with_dilation=None, norm_layer=None, groups=1, base_width=64):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = base_width
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width,
                            dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {"low_level": x1, "out": x4}

def _resnet(arch, block, layers, pretrained, replace_stride_with_dilation):
    model = ResNet(block, layers, replace_stride_with_dilation)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model

# ... (rest of the file remains the same)

def resnet101(pretrained=True, replace_stride_with_dilation=[False, False, True]):
    return _resnet('resnet101', Bottleneck, [3,4,23,3], pretrained, replace_stride_with_dilation)

# ---------------------------------------------------------------------
# ASPP, DeepLab heads (from _deeplab.py)
# ---------------------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        ]
        rates = list(atrous_rates)
        for r in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules)*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1))
    def forward(self, x):
        res = [conv(x) if not isinstance(conv, nn.AdaptiveAvgPool2d) else conv(x) for conv in self.convs]
        size = x.shape[-2:]
        res[-1] = F.interpolate(res[-1], size=size, mode='bilinear', align_corners=False)
        return self.project(torch.cat(res, dim=1))

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[6,12,18]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.aspp = ASPP(in_channels, aspp_dilate)
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1))
    def forward(self, features):
        low = self.project(features['low_level'])
        out = self.aspp(features['out'])
        out = F.interpolate(out, size=low.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low, out], dim=1))

class DeepLabV3(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x):
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear', align_corners=False)
        return x

# ---------------------------------------------------------------------
# Model factory (from modeling.py)
# ---------------------------------------------------------------------
def deeplabv3plus_resnet101(num_classes=19, output_stride=16, pretrained_backbone=True):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]; aspp_dilate=[12,24,36]
    else:
        replace_stride_with_dilation=[False, False, True]; aspp_dilate=[6,12,18]
    backbone = resnet101(pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    classifier = DeepLabHeadV3Plus(2048, 256, num_classes, aspp_dilate)
    return DeepLabV3(backbone, classifier)
