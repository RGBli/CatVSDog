from getdata import DogsVSCatsDataset as CatAndDogDataset
from network import Net
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import os

DATASET_DIR = './data'  # 数据集路径
MODEL_DIR = './model'  # 模型参数保存位置
WORKERS = 10  # PyTorch 读取数据线程数量
BATCH_SIZE = 16
LR = 0.001
EPOCH = 10
IMAGE_SIZE = 224  # 默认输入网络的图片大小

os.environ["CUDA_VISIBLE_DEVICES"] = 0

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = CatAndDogDataset(DATASET_DIR + "/train", 0, transform_train)
valset = CatAndDogDataset(DATASET_DIR + "/val", 1, transform_val)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=WORKERS)  # 用 PyTorch 的 DataLoader 类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果
val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=WORKERS)  # 用 PyTorch 的 DataLoader 类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果

model = Net()  # 实例化一个网络
model = model.cuda()  # 网络送入 GPU，即采用 GPU 计算，如果没有 GPU 加速，可以去掉".cuda()"
model = nn.DataParallel(model)
model.train()  # 网络设定为训练模式，有两种模式可选，.train() 和 .eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合

optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 实例化一个优化器，即调整网络参数，优化方式为 adam 方法
criterion = torch.nn.CrossEntropyLoss()  # 定义 loss 计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小


def train(epoch):
    print('\nEpoch: %d' % epoch)
    # 读取数据集中数据进行训练，因为 dataloader 的 batch_size 设置为16，所以每次读取的数据量为16，即 img 包含了16个图像，label 有16个
    for idx, (img, label) in enumerate(train_loader):  # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
        img, label = Variable(img).cuda(), Variable(label).cuda()  # 将数据放置在 PyTorch 的 Variable 节点中，并送入 GPU 中作为网络计算起点
        out = model(img)  # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的 forward() 方法
        loss = criterion(out,
                         label.squeeze())  # 计算损失，也就是网络输出值和实际 label 的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维 Tensor
        loss.backward()  # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
        optimizer.step()  # 优化采用设定的优化方法对网络中的各个参数进行调整
        optimizer.zero_grad()  # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都会累加
        print("Epoch:%d [%d|%d] loss:%f" % (epoch, idx, len(train_loader), loss.mean()))


def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(val_loader):
            img, label = Variable(img).cuda(), Variable(label).cuda()
            out = model(img)
            _, predicted = torch.max(out.data, 1)
            total += img.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))


if __name__ == '__main__':
    for epoch in range(EPOCH):
        train(epoch)
        val(epoch)
    torch.save(model.state_dict(), '{0}/model.pth'.format(MODEL_DIR))  # 训练所有数据后，保存网络的参数
