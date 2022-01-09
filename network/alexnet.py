from torch import nn
import torch

'''
reference https://github.com/sloth2012/AlexNet/blob/master/AlexNet.ipynb
https://github.com/dansuh17/alexnet-pytorch/blob/master/model.py
'''

class AlexNet(nn.Module):
    def __init__(self, n_class=2):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=5*5*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        
        self.layer8 = nn.Linear(in_features=4096, out_features=n_class)

        # initialize weights
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers and fully connected layers
        nn.init.constant_(self.layer2[0].bias, 1)
        nn.init.constant_(self.layer4[0].bias, 1)
        nn.init.constant_(self.layer5[0].bias, 1)
        nn.init.constant_(self.layer6[0].bias, 1)
        nn.init.constant_(self.layer7[0].bias, 1)
        nn.init.constant_(self.layer8.bias, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.layer8(self.layer7(self.layer6(x)))
        return x


if __name__ == '__main__':
    input = torch.rand(size=(4, 3, 224, 224))
    model = AlexNet()
    output = model(input)
    print(output.size())