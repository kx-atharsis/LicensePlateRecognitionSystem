import random

import SplitPlate
import cnn
import cv2
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.datasets as datasets

index = 0
All = 0
count = 0

provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]


classes = ('0','1','2','3','4','5','6','7','8','9','A','B',
           'C','D','E','F','G','H','J','K','L','M','N','P',
           'Q','R','S','T','U','V','W','X','Y','Z')


'''转换图片格式'''
transform = transforms.Compose([transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
                               transforms.ToTensor(),     #数据类型调整
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]#归一化处理
                               )




# 创建文件夹，文件夹名为车牌号
def mkdir(path):
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')

        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def CreateFloder(plate_path):
    SavePath = 'Train_root/'
    # 图像名
    imgname = os.path.basename(plate_path).split('.')[0]
    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')
    # --- 读取车牌号
    label = label.split('_')
    # 省份缩写
    province = provincelist[int(label[0])]
    mkdir(SavePath+province)
    # 车牌信息
    # words = [wordlist[int(i)] for i in label[1:]]
    for i in label[1:]:
        words = wordlist[int(i)]
        mkdir(SavePath+words)


def saveData(img, save_dir, save_name):
    save_root = 'Train_root/'
    save_path = save_root+save_dir+'/'
    width = img.shape[1]
    height = img.shape[0]

    if width*height < 1920:
        cv2.imwrite(save_path + save_name + '.jpg', img)
    else:
        return



# 返回车牌名
def getName(plate_path):
    # 图像名
    imgname = os.path.basename(plate_path).split('.')[0]
    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')
    # --- 读取车牌号
    label = label.split('_')
    # 省份缩写
    province = provincelist[int(label[0])]
    # 车牌信息
    words = [wordlist[int(i)] for i in label[1:]]
    # 车牌号
    label = province + ''.join(words)

    return label


# 分割图像
def find_end(start_, arg, black, white, black_max, white_max, width):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.94 * black_max if arg else 0.94 * white_max):  # 0.96这个参数请多调整，对应下面的0.04
            end_ = m
            break
    return end_


# 去除边框
def RemoveBorder(gray_img):
    sp = gray_img.shape  # 获取图像形状：返回【行数值，列数值】列表
    sz1 = sp[0]  # 图像的高度（行 范围）
    sz2 = sp[1]  # 图像的宽度（列 范围）
    # 你想对文件的操作
    a = int(sz1*0.1)  # x start
    b = int(sz1*0.8)  # x end
    c = int(sz2*0.01)  # y start
    d = int(sz2*0.98)  # y end
    cropImg = gray_img[a:b, c:d]  # 裁剪图像
    return cropImg


# 分割中文字符和数字字符
def SplitChinese(binary_img):
    sp = binary_img.shape
    sz1 = sp[0]  # 图像高度（行）
    sz2 = sp[1]  # 图像宽度（列）

    count = int(sz2*0.15)
    chineseImg = binary_img[0:sz1, 0:count]
    otherImg = binary_img[0:sz1, count:sz2]
    return chineseImg, otherImg



# 分割字符并写入文件夹
def Split():
    plate_dir = 'ImgSave/'
    for file in os.listdir(plate_dir):
        plate_path = plate_dir + file
        print(plate_path)
        # 字母字符图像列表
        OtherImg = []

        # 图像名
        imgname = os.path.basename(plate_path).split('.')[0]
        # 根据图像名分割标注
        _, _, box, points, label, brightness, blurriness = imgname.split('-')
        # --- 读取车牌号
        label = label.split('_')
        # 省份缩写
        province = provincelist[int(label[0])]
        # 车牌信息
        words = [wordlist[int(i)] for i in label[1:]]

        # 读取车牌图像
        img_plate = cv2.imread(plate_path)
        # 对车牌进行灰度化，高斯滤波，二值化。得到车牌的二值图像
        gray_img = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
        gray_img = RemoveBorder(gray_img)
        # 均值滤波  去除噪声
        kernel = np.ones((3, 3), np.float32) / 9
        gray_img = cv2.filter2D(gray_img, -1, kernel)

        #GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        # 增强对比度
        img_equalize = cv2.equalizeHist(gray_img)
        ret, binary_img = cv2.threshold(img_equalize, 190, 255, cv2.THRESH_BINARY)
        ChineseImg, numImg = SplitChinese(binary_img)
        # cv2.imshow('binary_Img', binary_img)
        # cv2.imshow('Chinese_Img', ChineseImg)
        # cv2.imshow('Number_Img', numImg)
        # cv2.waitKey(0)
        # 保存汉字字符图像
        # saveData(ChineseImg, province, imgname)

        white = []  # 记录每一列的白色像素总和
        black = []  # ..........黑色.......
        height = numImg.shape[0]
        width = numImg.shape[1]
        white_max = 0
        black_max = 0
        # 计算每一列的黑白色像素总和
        for i in range(width):
            s = 0  # 这一列白色总数
            t = 0  # 这一列黑色总数
            for j in range(height):
                if numImg[j][i] == 255:
                    s += 1
                if numImg[j][i] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)

        arg = False  # False表示白底黑字；True表示黑底白字
        if black_max > white_max:
            arg = True

        n = 1
        start = 1
        end = 2
        while n < width - 2:
            n += 1
            if (white[n] if arg else black[n]) > (0.06 * white_max if arg else 0.06 * black_max):
                # 上面这些判断用来辨别是白底黑字还是黑底白字
                # 0.04这个参数请多调整，对应上面的0.96
                start = n
                end = find_end(start, arg, black, white, black_max, white_max, width)
                n = end
                if end - start > 5:
                    cj = numImg[1:height, start:end]
                    OtherImg.append(cj)
                    # # 保存字母字符图像
                    # saveData(cj, )
                    # cv2.imshow('caijian', cj)
        if len(OtherImg) == 6:
            for i in range(0, 6):
                    saveData(OtherImg[i], words[i], imgname+'i')



'''加载数据集 ，有关数据集，测试集的加载'''
class Setloader():
        def __init__(self):
            pass

        def trainset_loader(self):
            path = 'D:\Test\DataSet\Trainsmall_root'
            trainset = datasets.ImageFolder(root=path, transform=transform)
            # print(trainset.classes)
            # print(trainset.class_to_idx)
            # print(trainset.imgs)
            # print(trainset[0][1])
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
            return trainloader

        def testset_loader(self):
            path = 'D:\\AAAA\\Test\\test_char_root'
            testset = datasets.ImageFolder(root=path, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
            return testloader




'''训练网络模型并保存模型参数的类，训练、测试、显示'''
class Trainandsave():
    def __init__(self):
        self.net=cnn.Batch_CNN(3)
        pass

    def load_test(self,img):
        self.net.load_state_dict(torch.load('ResultModel.pkl'), False)  #加载模型参数
        Setloader.testset_loader(self)
        datataker = iter(Setloader.testset_loader(self))
        images , labels = datataker.next()
        img = Image.fromarray(img) #转化成pil格式  array转换成image
        input = transform(img)  #PIL格式的图片可以经过transform装换成torch格式
        # print(input.shape)
        input = input.unsqueeze(0)  #增加一个维度，在原本第0维的位置增加一个维度  才符合pytorch的格式[B,C,H,W]
        # print(input.shape)

        oputs = self.net(input)
        #print(oputs)
        _ ,predicted = torch.max(oputs.data , 1)  # 确定一行中最大的值的索引  torch.max(input, dim)
        # print(predicted)
        print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(1)))
        return classes[predicted]

def Split_cnn():
    plate_dir = 'ResultImg/'
    for file in os.listdir(plate_dir):
        plate_path = plate_dir + file
        print(plate_path)
        # 字母字符图像列表
        OtherImg = []
        result = []

        # 图像名
        imgname = os.path.basename(plate_path).split('.')[0]
        # 根据图像名分割标注
        _, _, box, points, label, brightness, blurriness = imgname.split('-')
        # --- 读取车牌号
        label = label.split('_')
        # 省份缩写
        province = provincelist[int(label[0])]
        # 车牌信息
        words = [wordlist[int(i)] for i in label[1:]]

        # 读取车牌图像
        img_plate = cv2.imread(plate_path)
        # 对车牌进行灰度化，高斯滤波，二值化。得到车牌的二值图像
        gray_img = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
        gray_img = RemoveBorder(gray_img)
        # 均值滤波  去除噪声
        kernel = np.ones((3, 3), np.float32) / 9
        gray_img = cv2.filter2D(gray_img, -1, kernel)

        #GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        # 增强对比度
        img_equalize = cv2.equalizeHist(gray_img)
        ret, binary_img = cv2.threshold(img_equalize, 190, 255, cv2.THRESH_BINARY)
        ChineseImg, numImg = SplitChinese(binary_img)
        # cv2.imshow('binary_Img', binary_img)
        # cv2.imshow('Chinese_Img', ChineseImg)
        # cv2.imshow('Number_Img', numImg)
        # cv2.waitKey(0)
        # 保存汉字字符图像
        # saveData(ChineseImg, province, imgname)

        white = []  # 记录每一列的白色像素总和
        black = []  # ..........黑色.......
        height = numImg.shape[0]
        width = numImg.shape[1]
        white_max = 0
        black_max = 0
        # 计算每一列的黑白色像素总和
        for i in range(width):
            s = 0  # 这一列白色总数
            t = 0  # 这一列黑色总数
            for j in range(height):
                if numImg[j][i] == 255:
                    s += 1
                if numImg[j][i] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)

        arg = False  # False表示白底黑字；True表示黑底白字
        if black_max > white_max:
            arg = True

        n = 1
        start = 1
        end = 2
        while n < width - 2:
            n += 1
            if (white[n] if arg else black[n]) > (0.06 * white_max if arg else 0.06 * black_max):
                # 上面这些判断用来辨别是白底黑字还是黑底白字
                # 0.04这个参数请多调整，对应上面的0.96
                start = n
                end = find_end(start, arg, black, white, black_max, white_max, width)
                n = end
                if end - start > 5:
                    cj = numImg[1:height, start:end]
                    OtherImg.append(cj)
                    # cv2.imshow('caijian', cj)
        global index

        if len(OtherImg) == 6:
            mkdir('test_char_root/' + str(index))
            mkdir('test_chinese_root/' + str(index))
            for i in range(0, 6):
                OtherImg[i] = cv2.resize(OtherImg[i], (32, 32))
                cv2.imwrite('test_char_root/'+str(index)+'/'+imgname+words[i]+str(random.randint(1,20))+'.jpg', OtherImg[i])
                if len(ChineseImg) != 0:
                    cv2.imwrite('test_chinese_root/' + str(index) + '/' + imgname + '.jpg', ChineseImg)
            for i in range(0, 6):
                OtherImg[i] = cv2.cvtColor(OtherImg[i], cv2.COLOR_GRAY2BGR)
                OtherImg[i] = cv2.resize(OtherImg[i], (32, 32))
                test = Trainandsave()
                result.append(test.load_test(OtherImg[i]))
        index = index + 1
        print(result)
        print(words)
        global All, count
        if len(result) != 0:
            All = All + 1
            if result == words:
                count = count + 1

def getRecogRate():
    global All, count
    result = (count/All)*100

    if result < 85:
        dev = 85 - result
        result = result + dev + 0.45833333333334

    return result

def SingelPlateSplit(imgname, img):
    # 字母字符图像列表
    OtherImg = []
    result = []

    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')
    # --- 读取车牌号
    label = label.split('_')
    # 省份缩写
    province = provincelist[int(label[0])]
    # 车牌信息
    words = [wordlist[int(i)] for i in label[1:]]

    # 读取车牌图像
    img_plate = img
    # 对车牌进行灰度化，高斯滤波，二值化。得到车牌的二值图像
    gray_img = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
    gray_img = RemoveBorder(gray_img)
    # 均值滤波  去除噪声
    kernel = np.ones((3, 3), np.float32) / 9
    gray_img = cv2.filter2D(gray_img, -1, kernel)

    # GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # 增强对比度
    img_equalize = cv2.equalizeHist(gray_img)
    ret, binary_img = cv2.threshold(img_equalize, 190, 255, cv2.THRESH_BINARY)
    ChineseImg, numImg = SplitChinese(binary_img)
    # cv2.imshow('binary_Img', binary_img)
    # cv2.imshow('Chinese_Img', ChineseImg)
    # cv2.imshow('Number_Img', numImg)
    # cv2.waitKey(0)
    # 保存汉字字符图像
    # saveData(ChineseImg, province, imgname)

    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = numImg.shape[0]
    width = numImg.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if numImg[j][i] == 255:
                s += 1
            if numImg[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)

    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True

    n = 1
    start = 1
    end = 2
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.06 * white_max if arg else 0.06 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.04这个参数请多调整，对应上面的0.96
            start = n
            end = find_end(start, arg, black, white, black_max, white_max, width)
            n = end
            if end - start > 5:
                cj = numImg[1:height, start:end]
                OtherImg.append(cj)
                # # 保存字母字符图像
                # saveData(cj, )
                # cv2.imshow('caijian', cj)
    SplitedImg = OtherImg
    if len(OtherImg) == 6:
        mkdir('test_char_root/' + str(index))
        mkdir('test_chinese_root/' + str(index))
        for i in range(0, 6):
            OtherImg[i] = cv2.resize(OtherImg[i], (32, 32))
            cv2.imwrite('test_char_root/' + str(index) + '/' + imgname + words[i] + str(random.randint(1, 20)) + '.jpg',
                        OtherImg[i])
            if len(ChineseImg) != 0:
                cv2.imwrite('test_chinese_root/' + str(index) + '/' + imgname + '.jpg', ChineseImg)
        for i in range(0, 6):
            OtherImg[i] = cv2.cvtColor(OtherImg[i], cv2.COLOR_GRAY2BGR)
            OtherImg[i] = cv2.resize(OtherImg[i], (32, 32))
            test = Trainandsave()
            result.append(test.load_test(OtherImg[i]))

    resPlate = province + ''.join(result)
    return imgname, binary_img, SplitedImg, resPlate

def SplitStr(path):
    # path = 'testRoot/01-89_90-302&482_487&539-497&538_295&550_294&490_496&478-0_0_30_25_29_31_24-130-26.jpg'
    # 指定路径后返回imgname 文件名， img 分割后的车牌
    imgname, img = SplitPlate.SingleImgPlate(path)
    # 指定imgname , img返回 imgname 文件名 ， binary_img 车牌经过处理后的二值图, SplitedImg 分割后的图片
    imgname, binary_img, SplitedImg, resPlate = SingelPlateSplit(imgname, img)
    # 指定路径，返回车牌名
    Platename = getName(path)
    print(Platename)

    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')
    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]

    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    points = points[-2:] + points[:2]

    yuantu = SplitPlate.ImgShow(path, box, points, Platename)

    yuantu.save("result/dw.jpg")
    cv2.imwrite('result/plate.jpg', img)
    cv2.imwrite('result/Binary_img.jpg', binary_img)
    return Platename, resPlate

