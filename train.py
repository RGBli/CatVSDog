import torchvision
from dataset import CatAndDogDataset
import torch.utils.data
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss import SmoothedBCEWithLogitsLoss

# 数据集路径
DATASET_DIR = 'tinydata/'
# 模型参数保存路径
MODEL_DIR = 'model/'
# 日志保存路径
LOG_DIR = "log/"
# 每批读入的数据数量
BATCH_SIZE = 16
# 学习率
LR = 0.001
# 训练轮数
EPOCH = 30
# 默认输入网络的图片大小
IMAGE_SIZE = 224
# 类别数
N_CLASS = 2

# 有 GPU 则使用第一块，否则使用 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建 tensorboard writer 实例，用于记录训练曲线
summary_writer = SummaryWriter(LOG_DIR)

# 定义训练集的变换
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 定义验证集的变换
transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = CatAndDogDataset(DATASET_DIR + "train/", transform_train)
valset = CatAndDogDataset(DATASET_DIR + "val/", transform_val)
# 用 PyTorch 的 DataLoader 类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=4)

# 使用官方实现的经过预训练的 ResNet101 模型来实例化网络，为了保证性能并没有使用 network.py 中的网络
model = torchvision.models.resnet101(pretrained=True)
# 为了使得 ResNet101 能够应用在2分类问题上，添加了一个全连接层
model.fc = torch.nn.Linear(2048, N_CLASS)
model.to(device)

# 实例化一个优化器，即调整网络参数，优化方式为 adam 方法
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# 定义 scheduler，用于动态修正学习率，每训练10轮学习率降为之前的1/2
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# 定义 loss 计算方法
criterion = SmoothedBCEWithLogitsLoss()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    # 网络设定为训练模式，有两种模式可选，.train() 和 .eval()，训练模式和评估模式，区别就是训练模式采用了 dropout 策略，可以放置网络过拟合
    model.train()
    # 读取数据集中数据进行训练，因为 dataloader 的 batch_size 设置为16，所以每次读取的数据量为16，即 imgs 包含了16个图像，labels 有16个
    # 循环读取封装后的数据集，其实就是调用了数据集中的 __getitem__() 方法，只是返回数据格式进行了一次封装
    for idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        # 将数据放置在 PyTorch 的 Variable 节点中，并送入 GPU 中作为网络计算起点
        imgs = imgs.to(device)
        labels = labels.to(device)
        # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话每次计算梯度都会累加
        optimizer.zero_grad()
        # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的 forward() 方法
        out = model(imgs)
        # 计算损失，也就是网络输出值和实际 labels 的差异，差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维 Tensor
        loss = criterion(out, labels)
        # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
        loss.backward()
        # 采用设定的优化方法对模型中的各个参数进行调整
        optimizer.step()
        summary_writer.add_scalar("Train/Loss", loss.item(), (epoch - 1) * len(train_loader) + idx)
        print("Epoch:%d [%d|%d] loss:%f" % (epoch, idx + 1, len(train_loader), loss.item()))
    # 修正学习率
    scheduler.step()


def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    # 设定网络为评估模式
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            preds = torch.argmax(output, dim=1)
            total += imgs.size(0)
            correct += preds.data.eq(labels.data).cpu().sum()
    acc = (1.0 * correct.numpy()) / total
    summary_writer.add_scalar("Validation/Acc", acc, epoch)
    print("Epoch:%d acc: %f " % (epoch, acc))
    return acc


if __name__ == '__main__':
    max_acc = 0
    for epoch in range(1, EPOCH + 1):
        train(epoch)
        val_acc = val(epoch)
        # 保存 val_acc 最大的网络参数
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), '{0}/{1}_model.pth'.format(MODEL_DIR, val_acc))
    # 训练所有数据后，保存最后一轮的网络参数
    torch.save(model.state_dict(), '{0}/model.pth'.format(MODEL_DIR))
