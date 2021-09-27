import os
import torchvision.transforms.functional as TF
import random
import cv2
import matplotlib.pyplot as plt
import torch


def readImageName(input_file):
    fileList = os.listdir(input_file)
    baseNames = []
    for i in fileList:
        baseName = os.path.basename(i)
        baseName_list0 = baseName.split(".")
        baseName_0 = baseName_list0[0]
        baseName_list1 = baseName_0.split("_")
        baseName = baseName_list1[0]
        baseNames.append(baseName)

    return list(set(baseNames))  # 去掉重复图像名


def read0_5CDImageName(input_file):
    fileList = os.listdir(input_file)
    baseNames = []
    for i in fileList:
        baseName = os.path.basename(i)  # 00_0_1.png
        baseName_list0 = baseName.split(".")  # 00_0_1, png
        baseName_0 = baseName_list0[0]  # 00_0_1
        baseName_list1 = baseName_0.split("_")  # 00 ,0  , 1
        baseName_1 = baseName_list1[0] + "_" + baseName_list1[1]  # 00_0
        baseNames.append(baseName_1)
    return list(set(baseNames))


"""用于语义分割数据的数据增广"""


def roate(image, mask):
    """图像随机旋转，概率 50%"""
    if random.random() > 0.5:
        angle = random.randint(-20, 20)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    return image, mask


def vflipAndhflip(image, mask):
    """垂直反转"""
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    return image, mask


def pad(image, mask):
    if random.random() > 0.5:
        image = TF.pad(image, 10)
        mask = TF.pad(mask, 10)
    return image, mask


def data_transfomr_pipline(image, mask, pipline=None, size=224):
    image = TF.to_pil_image(image)
    mask = TF.to_pil_image(mask)

    for i in pipline:
        if random.random()>0.4:
            break
        if i == "roate":
            image, mask = roate(image, mask)
        elif i == "vflipAndhflip":
            image, mask = vflipAndhflip(image, mask)
        elif i == "pad":
            image, mask = pad(image, mask)
        else:
            break
    image = TF.resize(image, 224)
    mask = TF.resize(mask, 224)
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    image = TF.normalize(image, mean=[0.33, 0.34, 0.30], std=[0.17, 0.165, 0.17])

    return image, mask


"""用于cd dataset的数据增广"""


def cd_roate(image1, image2, mask):
    """图像随机旋转，概率 50%"""
    if random.random() > 0.5:
        angle = random.randint(-20, 20)
        image1 = TF.rotate(image1, angle)
        image2 = TF.rotate(image2, angle)
        mask = TF.rotate(mask, angle)
    return image1, image2, mask


def cd_vflipAndhflip(image1, image2, mask):
    """垂直反转"""
    if random.random() > 0.5:
        image1 = TF.vflip(image1)
        image2 = TF.vflip(image2)
        mask = TF.vflip(mask)
    if random.random() > 0.5:
        image1 = TF.hflip(image1)
        image2 = TF.hflip(image2)
        mask = TF.hflip(mask)

    return image1, image2, mask


def cd_pad(image1, image2, mask):
    if random.random() > 0.5:
        image1 = TF.pad(image1, 10)
        image2 = TF.pad(image2, 10)
        mask = TF.pad(mask, 10)
    return image1, image2, mask


def cd_data_transfomr_pipline(image1, image2, mask, pipline=None, size=224):
    if pipline is None:
        pipline = []
    image1 = TF.to_pil_image(image1)
    image2 = TF.to_pil_image(image2)
    mask = TF.to_pil_image(mask)
    for i in pipline:
        if i == "roate":
            image1, image2, mask = cd_roate(image1, image2, mask)
        elif i == "vflipAndhflip":
            image1, image2, mask = cd_vflipAndhflip(image1, image2, mask)
        elif i == "pad":
            image1, image2, mask = cd_pad(image1, image2, mask)
        else:
            break
    image1 = TF.resize(image1, 224)
    image2 = TF.resize(image2, 224)
    mask = TF.resize(mask, 224)
    image1 = TF.to_tensor(image1)
    image2 = TF.to_tensor(image2)
    mask = TF.to_tensor(mask)
    image1 = TF.normalize(image1, mean=[0.33, 0.34, 0.30], std=[0.17, 0.165, 0.17])
    image2 = TF.normalize(image2, mean=[0.33, 0.34, 0.30], std=[0.17, 0.165, 0.17])
    return image1, image2, mask


def get_one_hot(label, N):
    """将label转换成one-hot编码"""
    size = list(label.size())
    label = label.view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸

    return ones.view(*size)  # (h,w,N)


if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\dell\Desktop\Code\MyCDCode\data\gaofen_tiny\images\train\1_1.png")
    label = cv2.imread(r"C:\Users\dell\Desktop\Code\MyCDCode\data\gaofen_tiny\gt\train\1_1_label.png")
    image, mask = roate(image, label)
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()
