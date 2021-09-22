import csv

import torchvision.transforms as transforms
from get_data import CatAndDogDataset
from network import Net
import torch
import torch.utils.data
from torch.autograd import Variable

DATASET_DIR = './data'  # 数据集路径
MODEL_FILE = './model/model.pth'  # 模型保存路径
IMAGE_SIZE = 224  # 默认输入网络的图片大小

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset = CatAndDogDataset(DATASET_DIR + "/test/", transform_test)
test_loader = torch.utils.data.DataLoader(testset, )
model = Net()
model.to(device)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()
results = []


def test():
    with torch.no_grad():
        for image, label in test_loader:
            image = Variable(image.to(device))
            out = model(image)
            label = label.numpy().tolist()
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.data.cpu().numpy().tolist()
            # 第一列是 label，第二列是预测值
            results.extend([[i, ";".join(str(j))] for (i, j) in zip(label, predicted)])

        for result in results:
            print(result)


if __name__ == '__main__':
    test()
