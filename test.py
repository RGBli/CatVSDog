import torchvision.models
import torchvision.transforms as transforms
from dataset import CatAndDogDataset
import torch
import torch.utils.data
from torch.autograd import Variable

# 数据集路径
DATASET_DIR = './data'
# 模型保存路径
MODEL_FILE = './model/model.pth'
# 默认输入网络的图片大小
IMAGE_SIZE = 224
# 测试集的 batch_size
BATCH_SIZE = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset = CatAndDogDataset(DATASET_DIR + "/test/", transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
model = torchvision.models.resnet101()
model.fc = torch.nn.Linear(2048, 2)
model.to(device)
model.load_state_dict(torch.load(MODEL_FILE))


def test():
    correct = 0
    results = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = Variable(imgs.to(device))
            out = model(imgs)
            labels = labels.numpy().tolist()
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.data.cpu().numpy().tolist()
            # 第一列是 labels，第二列是预测值
            results.extend([[i, ";".join(str(j))] for (i, j) in zip(labels, predicted)])

        for result in results:
            if result[0] == int(result[1]):
                correct += 1
            print(result)

        print("Acc: ", correct / len(results))


if __name__ == '__main__':
    test()
