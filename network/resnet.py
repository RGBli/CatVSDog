import torch.nn as nn
import torch
import math
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


'''
reference https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/resnet.py
https://blog.csdn.net/jiangpeng59/article/details/79609392
'''


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# BasicBlock is used by ResNet18/32
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)
        return out


# Bottleneck is used by ResNet50/101/152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_class=2):
        self.inplanes = 64
        super().__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(m)

        self.layer2 = self._make_layer(block, 64, layers[0])
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, n_class))
            ])
        )

        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


if __name__ == '__main__':
    input = torch.rand(size=(4, 3, 224, 224))
    model = resnet50()
    output = model(input)
    print(output.size())
