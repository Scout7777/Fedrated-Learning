import torch.nn as nn
import torch
import torch.nn.functional as F


class SimpleMedicalNet(nn.Module):
    """简单的医学影像二分类网络"""

    def __init__(self):
        super(SimpleMedicalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_input_size = 16
        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.global_avg_pool(x)  # 输出形状: [batch, 16, 1, 1]
        x = x.view(-1, self.fc_input_size)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        #6组3个5*5的卷积核，得到6个二维矩阵特征图
        self.conv1 = nn.Conv2d(3, 6, 5)
        #下采样和特征增强，滑动窗口大小2*2，步长2，取其中最大值作为像素
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #假设输入图像的大小是32*32，经过两次卷积和池化操作是5*5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        #卷积后池化2次
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #展成1维
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)