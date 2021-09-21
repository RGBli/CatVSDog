import csv

import torchvision.transforms as transforms

from getdata import DogsVSCatsDataset as CatAndDogDataset
from network import Net
import torch
import torch.utils.data
from torch.autograd import Variable
import os

DATASET_DIR = './data'  # 数据集路径
MODEL_FILE = './model/model.pth'  # 模型保存路径
N = 10
IMAGE_SIZE = 224  # 默认输入网络的图片大小

os.environ["CUDA_VISIBLE_DEVICES"] = 0

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset = CatAndDogDataset(DATASET_DIR + "/test", 2, transform_test)
test_loader = torch.utils.data.DataLoader(testset, )
model = Net()
model.cuda()
model.load_state_dict(torch.load('model/model.pth'))
model.eval()
results = []


def test():
    with torch.no_grad():
        for image, label in test_loader:
            image = Variable(image.cuda())
            out = model(image)
            label = label.numpy().tolist()
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.data.cpu().numpy().tolist()
            results.extend([[i, ";".join(str(j))] for (i, j) in zip(label, predicted)])

    eval_csv = os.path.join(os.path.expanduser('.'), 'benchmark', 'eval.csv')

    with open(eval_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        q = ("id", "label")
        writer.writerow(q)
        for x in results:
            writer.writerow(x)


if __name__ == '__main__':
    test()
