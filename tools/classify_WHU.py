#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from PIL import Image
import pytesseract
import cv2
from matplotlib import pyplot as plt
import shutil
import glob


list_name=[]
root_dir = "/media/tang/lzp/whu_buildings"       #输入根目录，在此根目录下将会存放不同分类的图像文件夹


def estimate(imgname):
    img = cv2.imread(imgname,0)
    # ret,thresh = cv2.threshold(img,0,230,cv2.THRESH_BINARY)
    height,width = img.shape
    # print(height,width)
    size = img.size
    # print("size:"+ str(size))
    non_black_num = cv2.countNonZero(img)
    result = non_black_num/size
    # print("白色占比："+str(result))
    return result

def makenewdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def my_move(srcfn, dstdir):  ##定义移动函数，参数为待移动的文件路径和目标目录
    if not os.path.isfile(srcfn):  ##判断文件是否存在
        print('srcfn error')

    else:
        srcdir, fn = os.path.split(srcfn)  ##分离绝对路径和相对路径，获得文件名

        if not os.path.exists(dstdir):  ##如果目录不存在，创建目录
            os.makedirs(dstdir)

        dstfn = dstdir + fn  ##生成目录下的文件名
        shutil.copyfile(srcfn, dstfn)  ##复制


def get_common_name(path):
    filelist = os.listdir(path)
    common_name_flag = ''
    common_name_max = 0
    for file in filelist:

        Olddir = os.path.join(path, file)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则进入文件夹内处理，进行递归，达到处理子文件夹内全部文件的目的
            # print(Olddir)
            get_common_name(Olddir)
            # continue
        else:
            # fns2 = glob.glob(path + '/*_1_label.png')  ##获取当前目录下所有2.png格式的文件

            filename = os.path.splitext(file)[0]  # 分离文件名与扩展名;得到文件名
            # print(filename)

            filetype = os.path.splitext(file)[1]  # 文件扩展名(本例中是.png)
            # print(filetype)
            filepath = os.path.join(path, filename + filetype)
            # print(filepath)
            common_name = filename          #适用于WHU
            if not common_name in list_name:
                list_name.append(common_name)
            # print("common name is " + common_name)

    # print(list_name)
    return list_name


def classify(img_path,label_path):

    for name in list_name:
        # print(name)

        fns = glob.glob(img_path + '/'+ name +'.png')
        fns_label = glob.glob(label_path+'/' + name + '.png')

        img_percent = estimate(fns_label[0])
        if img_percent < 0.05:
            dst_dir =  os.path.join(root_dir, 'less')  # 存放目标存在20%以内的图片
            # print(dst_dir)
       #elif img_percent < 0.5:
            #dst_dir =  os.path.join(root_dir, '20--50')  # 存放目标存在20%以内的图片
        #elif img_percent < 0.7:
            #dst_dir = os.path.join(root_dir,'50--70')
        else:
            dst_dir = os.path.join(root_dir,'more')




        for img_name in fns:
            dst_dir_img = os.path.join(dst_dir,'images','train/')
            print(img_name+"  move to :  "+dst_dir_img)
            my_move(img_name,dst_dir_img)

        for img_name in fns_label:
            dst_dir_gt = os.path.join(dst_dir, 'gt', 'train/')
            print(img_name + "  move to :  "+dst_dir_gt)
            my_move(img_name,dst_dir_gt)



    return 0

if __name__ == '__main__':
    label_path = os.path.join(root_dir,'gt','train')#输入存放label图像的文件路径
    img_path = os.path.join(root_dir,'images','train')       #输入存放原图的文件路径
    get_common_name(label_path)
    classify(img_path,label_path)
    # move_all(".\\gaofen_small")
