#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
from PIL import Image

def split(path = r"C:\Users\dell\Desktop\Code\MyCDCode\Data (2)"):
    filelist = os.listdir(path)
    for file in filelist:
        Olddir = os.path.join(path, file)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则进入文件夹内处理，进行递归，达到处理子文件夹内全部文件的目的
            # print(Olddir)
            split(Olddir)
            continue
        else:
            newpath = os.path.join(path,'new')  #创建 原路径+new 的文件路径以便存放新图片
            if not os.path.exists(newpath):
                os.mkdir(newpath)
                # print(newpath)
            filename = os.path.splitext(file)[0]  # 分离文件名与扩展名;得到文件名
            # print(filename)
            filetype = os.path.splitext(file)[1]  # 文件扩展名(本例中是.png)
            # print(filetype)
            filepath = os.path.join(path,filename+filetype)
            # print(filepath)
            img = Image.open(filepath)              #打开一张图片
            size = img.size                         #获取大小

            weight = int(size[0] // 2)              # 四等分
            height = int(size[1] // 2)
            for j in range(2):
                for i in range(2):
                    newpath = os.path.join(path,'new','{}{}_'.format(j,i)+filename + filetype)
                    # print(newpath)                                      #重新命名文件，命名格式：原路径+new+00、01、11、10+文件原名称
                    box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
                    region = img.crop(box)                              #分割图片
                    region.save(newpath)                                #保存图片

if __name__ == '__main__':
    split(r"E:\World_Dataset\china\images\train")
    split(r"E:\World_Dataset\china\gt\train")
