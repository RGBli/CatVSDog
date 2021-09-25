import torchvision
from dataset import CatAndDogDataset
import torch.utils.data
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 数据集路径
DATASET_DIR = 'tinydata'
# 模型参数保存路径
MODEL_DIR = './model'
# 日志保存路径
LOG_DIR = "./log"
# PyTorch 读取数据线程数量
WORKERS = 10
BATCH_SIZE = 16
LR = 0.001
EPOCH = 10
# 默认输入网络的图片大小
IMAGE_SIZE = 224

# 有 GPU 则使用第一块，否则使用 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建 tensorboard writer 实例
summary_writer = SummaryWriter(LOG_DIR)

# 定义训练集的变换
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 定义验证集的变换
transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = CatAndDogDataset(DATASET_DIR + "/train/", transform_train)
valset = CatAndDogDataset(DATASET_DIR + "/val/", transform_val)
# 用 PyTorch 的 DataLoader 类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=WORKERS)
val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=WORKERS)

# 使用官方实现的 ResNet50 网络来实例化网络，为了保证性能并没有使用 network.py 中的网络
# model = Net()
model = torchvision.models.resnet50()
model.to(device)
# 网络设定为训练模式，有两种模式可选，.train() 和 .eval()，训练模式和评估模式，区别就是训练模式采用了 dropout 策略，可以放置网络过拟合
model.train()

# 实例化一个优化器，即调整网络参数，优化方式为 adam 方法
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# 定义 loss 计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小
criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    # 读取数据集中数据进行训练，因为 dataloader 的 batch_size 设置为16，所以每次读取的数据量为16，即 img 包含了16个图像，label 有16个
    # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
    for idx, (img, label) in enumerate(train_loader):
        # 将数据放置在 PyTorch 的 Variable 节点中，并送入 GPU 中作为网络计算起点
        img, label = Variable(img).to(device), Variable(label).to(device)
        # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的 forward() 方法
        out = model(img)
        # 计算损失，也就是网络输出值和实际 label 的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维 Tensor
        loss = criterion(out, label.squeeze())
        # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
        loss.backward()
        # 优化采用设定的优化方法对网络中的各个参数进行调整
        optimizer.step()
        # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话每次计算梯度都会累加
        optimizer.zero_grad()
        summary_writer.add_scalar("Train/Loss", loss.item(), (epoch - 1) * len(train_loader) + idx)
        print("Epoch:%d [%d|%d] loss:%f" % (epoch, idx + 1, len(train_loader), loss.mean()))


def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(val_loader):
            img, label = Variable(img).to(device), Variable(label).to(device)
            out = model(img)
            _, predicted = torch.max(out.data, 1)
            total += img.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
    acc = (1.0 * correct.numpy()) / total
    summary_writer.add_scalar("Validation/Acc", acc, epoch)
    print("Acc: %f " % acc)


if __name__ == '__main__':
    for epoch in range(1, EPOCH + 1):
        train(epoch)
        val(epoch)
    # 训练所有数据后，保存网络的参数
    torch.save(model.state_dict(), '{0}/model.pth'.format(MODEL_DIR))
