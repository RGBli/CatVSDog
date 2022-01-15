import math
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
reference https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
https://blog.csdn.net/u014380165/article/details/75142664
https://blog.csdn.net/weixin_41798111/article/details/86494353
'''

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, base_blocks, in_planes, growth_rate, block):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, base_blocks)

    def _make_layer(self, block, in_planes, growth_rate, base_blocks):
        layers = []
        for i in range(base_blocks):
            layers.append(block(in_planes + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNetBC(nn.Module):
    def __init__(self, depth, n_class, growth_rate=12, reduction=0.5, bottleneck=True):
        super().__init__()
        in_planes = 2 * growth_rate
        base_blocks = (depth - 4) // 3
        if bottleneck == True:
            base_blocks //= 2
            block = BottleneckBlock
        else:
            block = BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 1st block
        self.block1 = DenseBlock(base_blocks, in_planes, growth_rate, block)
        in_planes = int(in_planes + base_blocks * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)))
        in_planes = math.floor(in_planes * reduction)

        # 2nd block
        self.block2 = DenseBlock(base_blocks, in_planes, growth_rate, block)
        in_planes = int(in_planes + base_blocks * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(
            math.floor(in_planes * reduction)))
        in_planes = math.floor(in_planes * reduction)

        # 3rd block
        self.block3 = DenseBlock(base_blocks, in_planes, growth_rate, block)
        in_planes = int(in_planes + base_blocks * growth_rate)

        # global average pooling and classifier
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(132*50*50, n_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                base_blocks = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / base_blocks))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, kernel_size=7, stride=1)
        # print(out.size())
        out = out.view(-1, 132*50*50)
        return self.fc(out)


if __name__ == '__main__':
    input = torch.rand(size=(4, 3, 224, 224))
    model = DenseNetBC(depth=40, n_class=2)
    output = model(input)
    print(output.size())