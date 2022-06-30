import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import os
import SplitPlate
import SplitStr

'''按下按钮后的界面展示'''


def show(pic, pic1, pic2, plateName, resPlate):
    clean()
    imgx = Image.open(pic)
    imgx.thumbnail((500, 500))
    imgx1 = Image.open(pic1)
    imgx1.thumbnail((200, 100))
    imgx2 = Image.open(pic2)
    imgx2.thumbnail((200, 100))

    photox = ImageTk.PhotoImage(imgx)
    global canvas
    canvas = tk.Canvas(app, width=500, height=430)
    canvas.create_image(0, 0, image=photox, anchor='nw', tag='1')
    canvas.place(x=0, y=27)

    photox1 = ImageTk.PhotoImage(imgx1)
    global canvas1
    canvas1 = tk.Canvas(app, width=300, height=80)
    canvas1.create_image(0, 0, image=photox1, anchor='nw', tag='2')
    canvas1.place(x=527, y=27)
    text.insert(1.0, str(plateName))

    photox2 = ImageTk.PhotoImage(imgx2)
    global canvas2
    canvas2 = tk.Canvas(app, width=300, height=80)
    canvas2.create_image(0, 0, image=photox2, anchor='nw', tag='3')
    canvas2.place(x=527, y=299)
    text1.insert(1.0, str(resPlate))
    app.mainloop()


'''清除界面中的全部图片'''


def clean():
    text.delete(1.0, tk.END)
    text1.delete(1.0, tk.END)
    text2.delete(1.0, tk.END)
    canvas.delete("all")
    canvas1.delete("all")
    canvas2.delete("all")


'''集中测试'''


def sh():
    clean()
    SplitPlate.DataLoad()
    SplitStr.Split_cnn()
    res = SplitStr.getRecogRate()
    text2.insert(1.0, str(res))
    return 0


'''选择图片'''


def sh1():
    global pic
    pic = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
    print("pic=", pic)
    PlateName, resPlate = SplitStr.SplitStr(pic)
    if len(resPlate) == 1:
        resPlate = '识别出错'
    pic0 = 'result/dw.jpg'
    pic1 = 'result/plate.jpg'
    pic2 = 'result/Binary_img.jpg'
    show(pic0, pic1, pic2, PlateName, resPlate)
    return 0


def qvpath(path):
    g = []
    for dirpath, dirnames, filenames in os.walk(path):
        print("本文件路径:", dirpath, "文件夹名:", dirnames, "文件名:", filenames)
        for fi in filenames:
            g.append(dirpath + '\\' + fi)
    return g


# 前一张


def sh2():
    global pic
    if qvp.index(pic) - 1 >= 0:
        pic = qvp[qvp.index(pic) - 1]
    else:
        pic = qvp[len(qvp) - 1]
    PlateName, resPlate = SplitStr.SplitStr(pic)
    if len(resPlate) == 1:
        resPlate = '识别出错'
    pic0 = 'result/dw.jpg'
    pic1 = 'result/plate.jpg'
    pic2 = 'result/Binary_img.jpg'
    show(pic0, pic1, pic2, PlateName, resPlate)
    return 0


# 后一张


def sh3():
    global pic
    if qvp.index(pic) + 1 < len(qvp):
        pic = qvp[qvp.index(pic) + 1]
    else:
        pic = qvp[0]
    PlateName, resPlate = SplitStr.SplitStr(pic)
    if len(resPlate) == 1:
        resPlate = '识别出错'
    pic0 = 'result/dw.jpg'
    pic1 = 'result/plate.jpg'
    pic2 = 'result/Binary_img.jpg'
    show(pic0, pic1, pic2, PlateName, resPlate)
    return 0


if __name__ == '__main__':
    qvp = qvpath('D:\\code\\Python\\LicensePlateRecognitionSystem\\TeacherTest')
    i = 0
    while i < len(qvp):
        qvp[i] = qvp[i].replace('\\', '/')
        i += 1
    # 标题
    app = tk.Tk()
    app.title("车牌识别")
    app.geometry("1024x680")
    # 标签
    lb = tk.Label(app, text="定位车牌", font=("宋体", 12))
    lb1 = tk.Label(app, text="车牌分割", font=("宋体", 12))
    lb2 = tk.Label(app, text="车牌原字", font=("宋体", 12))
    lb3 = tk.Label(app, text="车牌处理", font=("宋体", 12))
    lb4 = tk.Label(app, text="识别结果", font=("宋体", 12))
    lb5 = tk.Label(app, text="集中测试", font=("宋体", 12))
    lb.place(x=190, y=0)
    lb1.place(x=550, y=0)
    lb2.place(x=550, y=136)
    lb3.place(x=550, y=272)
    lb4.place(x=550, y=408)
    lb5.place(x=710, y=476)
    # 按钮
    bu = tk.Button(text='集中测试', command=sh, fg='green', font=("宋体", 12))
    bu1 = tk.Button(text='选择图片', command=sh1, fg='green', font=("宋体", 12))

    bu2 = tk.Button(text='-前一张-', command=sh2, fg='green', font=("宋体", 12))
    bu3 = tk.Button(text='-后一张-', command=sh3, fg='green', font=("宋体", 12))

    bu.place(x=583, y=544)
    bu1.place(x=819, y=544)

    bu2.place(x=583, y=612)
    bu3.place(x=819, y=612)

    # 文本框（输出）
    text = tk.Text(app, height=1, fg='black', width=10)
    text1 = tk.Text(app, height=1, fg='black', width=10)
    text2 = tk.Text(app, height=1, fg='black', width=10)
    text.place(x=551, y=163)
    text1.place(x=551, y=435)
    text2.place(x=709, y=503)
    # 图片
    '''img = Image.open('test1.jpg')
    img.thumbnail((500, 500))
    img1 = Image.open('test2.jpg')
    img1.thumbnail((200, 100))
    img2 = Image.open('test3.jpg')
    img2.thumbnail((200, 100))'''

    canvas = tk.Canvas(app, width=500, height=430)

    canvas.place(x=10, y=30)

    canvas1 = tk.Canvas(app, width=300, height=80)

    canvas1.place(x=551, y=27)

    canvas2 = tk.Canvas(app, width=300, height=80)

    canvas2.place(x=551, y=299)

    app.mainloop()
