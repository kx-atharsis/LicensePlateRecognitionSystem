"""加载数据训练集、测试集，并训练，得到训练模型，并测试模型精度"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import cnn
import SplitStr
import SplitPlate

# 超参数(Hyperparameters)
batch_size = 64  # 批处理数量，每批处理64张图像
learning_rate = 0.001
num_epoches = 60

data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

img_transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
    # 将数据转化为Tensor类型
    transforms.ToTensor(),
    # 归一化。注意，下面的mean和std分别为数据集中所有图像数据的均值和标准差，以实际数据集为准，目前给出的值仅为默认初值
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 将rgb图片的每个分量转换到[-1,1]之间
])


def trainset_loader():
    path = 'D:\\Test\\DataSet\\Trainsmall_root'
    trainset = datasets.ImageFolder(root=path, transform=img_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    return trainset, trainloader


def testset_loader():
    path = 'D:\\LicensePlateRecognitionSystem\\DataSet\\test_root'
    testset = datasets.ImageFolder(root=path, transform=img_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return testset, testloader


train_datafile, train_dataloader = trainset_loader()
test_datafile, test_dataloader = testset_loader()

model = cnn.Batch_CNN(3)

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)

if __name__ == '__main__':  # 训练数据
    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        model.train()  # 开启训练模式，即启用batch normalization和drop out
        for data in train_dataloader:  # data为train_loader中的一个批次样本
            img, label = data
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            # ==========forward====================
            out = model(img)
            loss = criterion(out, label)
            # ==========backward===================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # =========record loss=================
            train_loss += loss.data / len(train_datafile)

            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            train_acc += num_correct.numpy() / len(train_datafile)

        # 输出阶段训练结果
        print('*' * 10)
        print('epoch: {}, train loss: {:.4f}, train acc: {:.4f}'.format(epoch + 1, train_loss, train_acc))

        # 测试数据
        model.eval()  # 让model变为测试模式，网络会沿用batch normalization的值，但不使用drop out
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for data in test_dataloader:
                img, label = data
                if torch.cuda.is_available():
                    img = Variable(img).cuda()
                    label = Variable(label).cuda()
                else:
                    img = Variable(img)
                    label = Variable(label)
                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.data * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.data
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss.numpy() / (len(test_datafile)),
                                                      eval_acc.numpy() / (len(test_datafile))))

    torch.save(model.state_dict(), 'ResultModel.pkl')  # 训练结束，保存模型参数
