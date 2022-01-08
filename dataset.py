import os
import torch
from torch.utils.data import Dataset
from PIL import Image


# 新建一个数据集类，并且需要继承 PyTorch 中的 tinydata.Dataset 父类
class CatAndDogDataset(Dataset):
    # 默认构造方法，传入数据集类别（训练或测试），以及数据集路径
    def __init__(self, dir, transform):
        self.transform = transform
        # 新建一个 image list，用于存放图片路径，注意是图片路径
        self.imgs = []
        # 新建一个 label list，用于存放图片对应猫或狗的标签，其中数值0表示猫，1表示狗
        self.labels = []

        # 遍历 dir 文件夹，将图片路径和文件名添加至 image list，label 添加至 label list
        for file in os.listdir(dir):
            self.imgs.append(dir + file)
            # 分割文件名，"cat.0.jpg" 将分割成 "cat",".","jpg" 3个元素
            name = file.split(sep='.')
            # label 采用 one-hot 编码，"10"表示猫，"01"表示狗，任何情况只有一个位置为"1"，在采用 CrossEntropyLoss() 计算 Loss 情况下，label
            # 只需要输入"1"的索引，即猫应输入0，狗应输入1
            # 图片为猫，label 为0，图片为狗，label 为1
            if name[0] == 'cat':
                self.labels.append(0)
            else:
                self.labels.append(1)

    # 重写 Dataset 父类方法，获取数据集中数据内容
    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        label = self.labels[index]
        return self.transform(img), self.one_hot(label, 2)

    # 重写 Dataset 父类方法，获取数据数量
    def __len__(self):
        return len(self.imgs)

    # 将标签转为 one-hot 形式
    @staticmethod
    def one_hot(label, n_class=2):
        return torch.nn.functional.one_hot(torch.tensor(label), n_class).float()
