import torchvision.transforms as transforms

from getdata import DogsVSCatsDataset as CatAndDogDataset
from network import Net
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata

DATASET_DIR = './data/test/'                    # 数据集路径
MODEL_FILE = './model/model.pth'                # 模型保存路径
N = 10

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset = DogCat('./data/test1',transform=transform_test,train=False,test=True)
testloader=torch.utils.data.DataLoader(testset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)
model=resnet101(pretrained=True)
model.fc=nn.Linear(2048,2)
model.load_state_dict(torch.load('ckp/model.pth'))
model.cuda()
model.eval()
results=[]

def test():
    # setting model
    model = Net()                                       # 实例化一个网络
    model.cuda()                                        # 送入 GPU，利用 GPU 计算
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(MODEL_FILE))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要 dropout

    # get data
    files = random.sample(os.listdir(DATASET_DIR), N)   # 随机获取 N 个测试图像
    imgs = []           # img
    imgs_data = []      # img data
    for file in files:
        img = Image.open(DATASET_DIR + file)            # 打开图像
        img_data = getdata.dataTransform(img)           # 转换成 torch tensor 数据

        imgs.append(img)                                # 图像 list
        imgs_data.append(img_data)                      # tensor list
    imgs_data = torch.stack(imgs_data)                  # tensor list 合成一个 4D tensor

    # calculation
    out = model(imgs_data)                              # 对每个图像进行网络计算
    out = F.softmax(out, dim=1)                         # 输出概率化
    out = out.data.cpu().numpy()                        # 转成 numpy 数据

    # print result
    for idx in range(N):
        plt.figure()
        if out[idx, 0] > out[idx, 1]:
            plt.suptitle('cat:{:.1%},dog:{:.1%}'.format(out[idx, 0], out[idx, 1]))
        else:
            plt.suptitle('dog:{:.1%},cat:{:.1%}'.format(out[idx, 1], out[idx, 0]))
        plt.imshow(imgs[idx])
    plt.show()


if __name__ == '__main__':
    test()


