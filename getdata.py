import os
import torch.utils.data as data
import random
from PIL import Image


class CatAndDogDataset(data.Dataset):  # 新建一个数据集类，并且需要继承 PyTorch 中的 data.Dataset 父类
    def __init__(self, dir, transform):  # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.transform = transform
        self.imgs = []  # 新建一个 image list，用于存放图片路径，注意是图片路径
        self.labels = []  # 新建一个 label list，用于存放图片对应猫或狗的标签，其中数值0表示猫，1表示狗

        for file in os.listdir(dir):  # 遍历 dir 文件夹
            self.imgs.append(dir + file)  # 将图片路径和文件名添加至 image list                  # 数据集增1
            name = file.split(sep='.')  # 分割文件名，"cat.0.jpg" 将分割成 "cat",".","jpg" 3个元素
            # label 采用 one-hot 编码，"1,0"表示猫，"0,1"表示狗，任何情况只有一个位置为"1"，在采用 CrossEntropyLoss() 计算 Loss 情况下，label
            # 只需要输入"1"的索引，即猫应输入0，狗应输入1
            if name[0] == 'cat':
                self.labels.append(0)  # 图片为猫，label 为0
            else:
                self.labels.append(1)  # 图片为狗，label 为1

    def __getitem__(self, item):  # 重载 data.Dataset 父类方法，获取数据集中数据内容
        img = Image.open(self.imgs[item])  # 打开图片
        label = self.labels[item]  # 获取 image 对应的 label
        return self.transform(img), label  # 将 image 和 label 转换成 PyTorch 形式并返回

    def __len__(self):
        return len(self.imgs)
