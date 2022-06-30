"""用于车牌定位并分割出车牌"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


# --- 绘制边界框


def DrawBox(im, box):
    draw = ImageDraw.Draw(im)
    draw.rectangle([tuple(box[0]), tuple(box[1])], outline="#FFFFFF", width=3)


# --- 绘制四个关键点


def DrawPoint(im, points):
    draw = ImageDraw.Draw(im)

    for p in points:
        center = (p[0], p[1])
        radius = 5
        right = (center[0] + radius, center[1] + radius)
        left = (center[0] - radius, center[1] - radius)
        draw.ellipse((left, right), fill="#FF0000")


# --- 绘制车牌
def DrawLabel(im, label):
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('simsun.ttc', 64)
    draw.text((30, 30), label, font=font)


# --- 图片可视化
def ImgShow(imgpath, box, points, label):
    # 打开图片
    im = Image.open(imgpath)
    DrawBox(im, box)
    DrawPoint(im, points)
    return im


# 车牌矫正
def transform(img, points, height, width):
    points1 = np.float32(points)
    points2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(points1, points2)
    processed = cv2.warpPerspective(img, M, (width, height))
    return processed


# 调整图像大小
def resize(img, height, width):
    img_size = (height, width)
    resized_img = cv2.resize(img, img_size)
    return resized_img


# 保存车牌图像, 修改保存路径，可保存到不同文件夹下
def ImgSave(imgpath, points, height, width):
    # 保存路径
    save_dir = 'ResultImg/'
    imgname = os.path.basename(imgpath).split('.')[0]
    # 打开图片
    im = cv2.imread(imgpath)
    # 进行倾斜矫正
    im = transform(im, points, height, width)
    # 调整图像大小
    im = resize(im, 220, 70)
    print(save_dir + imgname)
    # 保存车牌图片
    cv2.imwrite(save_dir + imgname + '.jpg', im)


# 用于集中测试的车牌定位分离
def DataLoad():
    # 车牌集路径
    testRoot = 'D:\\code\\Python\\LicensePlateRecognitionSystem\\TeacherTest\\TestRoot\\'
    for file in os.listdir(testRoot):
        imgpath = testRoot + file
        # 图像名
        imgname = os.path.basename(imgpath).split('.')[0]

        # 根据图像名分割标注
        _, _, box, points, label, brightness, blurriness = imgname.split('-')

        # --- 边界框信息
        box = box.split('_')
        box = [list(map(int, i.split('&'))) for i in box]
        width = box[1][0] - box[0][0]
        height = box[1][1] - box[0][1]
        print(imgpath)
        print(width)
        print(height)
        print(box)
        # --- 关键点信息
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        points = points[-2:] + points[:2]
        print(points)
        ImgSave(imgpath, points, height, width)


# 部分预处理：规范关键点、倾斜矫正、规范图像大小
def SingleImgPlate(path):
    imgpath = path
    # 图像名
    imgname = os.path.basename(imgpath).split('.')[0]

    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')

    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]
    width = box[1][0] - box[0][0]
    height = box[1][1] - box[0][1]
    print(imgpath)
    print(width)
    print(height)
    print(box)
    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    points = points[-2:] + points[:2]
    print(points)
    img = cv2.imread(imgpath)
    # 进行倾斜矫正
    img = transform(img, points, height, width)
    # 调整图像大小
    img = resize(img, 220, 70)

    return imgname, img
