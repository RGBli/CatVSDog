import math
import torch
import torch.nn as nn

__all__ = [
    'VGG', 'vgg11A', 'vgg11A_bn', 'vgg13B', 'vgg16C', 'vgg16D', 'vgg19E',
]

'''
reference https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
https://github.com/minar09/VGG16-PyTorch/blob/master/vgg.py
'''


class VGG(nn.Module):
    def __init__(self, features, n_class=2):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_class),
        )

        # Initialize weights

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)
        return x


def make_layer(cfg, type='E', batch_norm=False):
    cfg = cfg[type]
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if type == 'C' and i in [8, 12, 16]:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0)
                layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11A():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layer(cfg, 'A'))


def vgg11A_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layer(cfg, 'A', batch_norm=True))


def vgg13B():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layer(cfg, 'B'))


def vgg16C():
    """VGG 16-layer model (configuration "C")"""
    return VGG(make_layer(cfg, 'C'))


def vgg16D():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layer(cfg, 'D'))


def vgg19E():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layer(cfg, 'E'))


if __name__ == '__main__':
    input = torch.rand(size=(4, 3, 224, 224))
    model = vgg16C()
    output = model(input)
    print(output.size())
