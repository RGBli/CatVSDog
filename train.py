from getdata import DogsVSCatsDataset as CatAndDogDataset
from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

dataset_dir = './data/'             # 数据集路径
model_dir = './model/'               # 模型参数保存位置
workers = 10                        # PyTorch 读取数据线程数量
batch_size = 16                    
lr = 0.0001                        
nepoch = 10


def train():
    datafile = CatAndDogDataset('train', dataset_dir)                                                           # 实例化一个数据集
    dataloader = torch.utils.data.DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)     # 用 PyTorch 的 DataLoader 类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果

    print('Dataset loaded! Size of train set is {0}'.format(len(datafile)))

    model = Net()                       # 实例化一个网络
    model = model.cuda()                # 网络送入 GPU，即采用 GPU 计算，如果没有 GPU 加速，可以去掉".cuda()"
    model = nn.DataParallel(model)
    model.train()                       # 网络设定为训练模式，有两种模式可选，.train() 和 .eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)         # 实例化一个优化器，即调整网络参数，优化方式为 adam 方法

    criterion = torch.nn.CrossEntropyLoss()                         # 定义 loss 计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小

    cnt = 0                             # 训练图片数量
    for epoch in range(nepoch):
        # 读取数据集中数据进行训练，因为 dataloader 的 batch_size 设置为16，所以每次读取的数据量为16，即 img 包含了16个图像，label 有16个
        for img, label in dataloader:                                           # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
            img, label = Variable(img).cuda(), Variable(label).cuda()           # 将数据放置在 PyTorch 的 Variable 节点中，并送入 GPU 中作为网络计算起点
            out = model(img)                                                    # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的 forward() 方法
            loss = criterion(out, label.squeeze())      # 计算损失，也就是网络输出值和实际 label 的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维 Tensor
            loss.backward()                             # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()                            # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()                       # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            cnt += 1

            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))          # 打印一个 batch size 的训练结果

    torch.save(model.state_dict(), '{0}/model.pth'.format(model_dir))            # 训练所有数据后，保存网络的参数


if __name__ == '__main__':
    train()










