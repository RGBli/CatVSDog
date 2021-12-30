import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


# 新建一个网络类，就是需要搭建的网络，必须继承 PyTorch 的 nn.Module 父类
class Net(nn.Module):
    # 构造函数，用于设定网络层
    def __init__(self):
        # 标准语句，继承父类的构造函数
        super().__init__()
        # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding 大小1，其他参数默认
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), padding=1)
        # 第二个卷积层，输入通道数16，输出通道数16，卷积核大小3×3，padding 大小1，其他参数默认
        self.conv2 = torch.nn.Conv2d(16, 16, (3, 3), padding=1)
        # 第一个全连层，线性连接，输入节点数56×56×16，输出节点数128
        self.fc1 = nn.Linear(56 * 56 * 16, 128)
        # 第二个全连层，线性连接，输入节点数128，输出节点数64
        self.fc2 = nn.Linear(128, 64)
        # 第三个全连层，线性连接，输入节点数64，输出节点数2
        self.fc3 = nn.Linear(64, 2)

    # 重写父类 forward 方法，即前向计算，通过该方法获取网络输入数据后的输出值
    def forward(self, x):
        # 第一次卷积
        x = self.conv1(x)
        # 第一次卷积结果经过 ReLU 激活函数处理
        x = F.relu(x)
        # 第一次池化，池化大小2×2，方式 Max pooling
        x = F.max_pool2d(x, 2)

        # 第二次卷积
        x = self.conv2(x)
        # 第二次卷积结果经过ReLU激活函数处理
        x = F.relu(x)
        # 第二次池化，池化大小2×2，方式 Max pooling
        x = F.max_pool2d(x, 2)

        # 由于全连层输入的是一维张量，因此需要对输入的[50×50×16]格式数据排列成[40000×1]形式
        x = x.view(x.size()[0], -1)
        # 第一次全连，ReLU 激活
        x = F.relu(self.fc1(x))
        # 第二次全连，ReLU 激活
        x = F.relu(self.fc2(x))
        # 第三次激活
        y = self.fc3(x)

        return y
