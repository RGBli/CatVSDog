from torch import nn


# 添加 label smoothing 的 BCEWithLogitsLoss 损失
class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 0.05

    def forward(self, input, target):
        target = target * (1 - self.smooth) + 0.5 * self.smooth
        bce = nn.BCEWithLogitsLoss()
        return bce.forward(input, target)
