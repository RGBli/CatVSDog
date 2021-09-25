import torchvision.models
import torchvision.transforms as transforms
from dataset import CatAndDogDataset
import torch
import torch.utils.data
from torch.autograd import Variable

# 数据集路径
DATASET_DIR = 'tinydata'
# 模型保存路径
MODEL_FILE = './model/model.pth'
# 默认输入网络的图片大小
IMAGE_SIZE = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset = CatAndDogDataset(DATASET_DIR + "/test/", transform_test)
test_loader = torch.utils.data.DataLoader(testset, )
model = torchvision.models.resnet50()
model.to(device)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()
results = []


def test():
    correct = 0
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
            if result[0] == int(result[1]):
                correct += 1
            print(result)

        print("Acc: ", correct / len(results))


if __name__ == '__main__':
    test()
