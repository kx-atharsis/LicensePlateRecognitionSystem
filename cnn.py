"""cnn LeNet-5神经网络模型"""

import torch.nn as nn


# 搭建神经网络
class Batch_CNN(nn.Module):
    def __init__(self, c):  # 初始化
        # 在单继承中 super 主要是用来调用父类的方法的。
        # super().__init__()的作用，就是执行父类的构造函数，使能够调用父类的属性。
        super(Batch_CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 该函数将按照参数传递的顺序将其依次添加到处理模块中
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,[dilation, groups, bias])，各参数如下：
            # in_channels对应输入数据体的深度/通道数
            # out_channels对应输出数据体的深度（特征图的数量），该参数定义 滤波器（卷积核）的数量。
            # kernel_size表示滤波器(卷积核)的大小，使用一个数字表示相同高宽的卷积核，或不同数字表示高宽不等的卷积核，如kernel_size=(3,2)
            # stride表示滑动步长，默认stride=1
            # padding为周围0填充行数，padding=0(默认)为不填充
            # bias是一个布尔值，默认bias=True，表示使用偏移置
            # dilation表示卷积对于输入数据体的空间间隔，默认dilation=1
            # groups表示输出数据体深度上和输入数据体深度上的联系，默认groups=1,也就是所有输出和输入都是关联的。
            # 第一次卷积
            nn.Conv2d(c, 6, kernel_size=5),  # 输出特征图个数(深度)16,大小26*26
            # 批量标准化操作
            nn.BatchNorm2d(6),  # Batch Normalization强行将数据拉回到均值为0，方差为1的正态分布上，参数为 特征图的数量
            # 激活函数负责将来自节点的加权输入转换为该输入的节点或输出的激活
            nn.ReLU(inplace=True),  # inplace=True节省内(显)存空间，省去反复申请和释放内存的时间，但会对输入的变量进行覆盖，类似地址传递
            # 第一次池化
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出特征图个数为32 MaxPool2d 池化
        )

        self.layer2 = nn.Sequential(
            # 第二次卷积
            nn.Conv2d(6, 16, kernel_size=5),  # 输出特征图个数为32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 第二次池化
            # kernel_size ：表示做最大池化的窗口大小
            # stride ：步长
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出特征图个数为32 MaxPool2d 池化
        )

        self.fc = nn.Sequential(
            # 第一次线性
            nn.Linear(16 * 12 * 12, 1024),
            nn.ReLU(inplace=True),
            # 第二次线性
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            # 第三次线性
            nn.Linear(128, 34)
        )

    # 前向传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 16 * 12 * 12)
        x = self.fc(x)
        return x
