from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from tools import utils


class CDImgDataset(Dataset):
    def __init__(self, input_file, transform=None, mode="train", label_norm=1):

        self.label_norm = label_norm
        self.mode = mode  # train or test
        self.x, self.y, self.imageName = self.parseInput(input_file)  # 处理输入路径

        self.transform = transform

        self.image_files = os.listdir(self.x)
        self.label_files = os.listdir(self.y)

    def parseInput(self, input_file):
        x = os.path.join(input_file, "images", self.mode)  # input/images/train or test
        y = os.path.join(input_file, "gt", self.mode)  # input/gt/train or test    可以根据数据格式进行修改

        imageNames = utils.read0_5CDImageName(x)  # 用来读取图像名

        return x, y, imageNames

    def __len__(self):
        return len(self.imageName)

    def __getitem__(self, index):
        result = {}
        base_name = self.imageName[index]
        # print(os.path.join(self.x, str(base_name) + "_1.png"))
        image1 = cv2.imread(os.path.join(self.x, base_name + "_1.png"))  # imagename_1.png
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # cv2 读取下来是BGR，所以需要强制更改通道顺序为RGB
        image2 = cv2.imread(os.path.join(self.x, base_name + "_2.png"))  # imagename_2.png
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        label_path = base_name
        label = cv2.imread(os.path.join(self.y, label_path + "_change.png"), cv2.IMREAD_GRAYSCALE)  # 【h，w】
        # label = cv2.imread(os.path.join(self.y, str(label_path) + "_change.png"))
        # label = TF.to_grayscale(label)
        label = label // self.label_norm
        if self.mode == "train":
            # 数据增广操作，修改pipline即可修改增广操作与顺序
            image1, image2, label = utils.cd_data_transfomr_pipline(image1, image2, label,
                                                                    pipline=["roate", "vflipAndhflip", "pad"], size=224)
            label = np.where(label != 0, 1, 0)  # 插值之后label会存在0-255之间的值
            result["image1"] = image1
            result["image2"] = image2
            result["label"] = label
            # print(torch.max(label))
            return result
        elif self.mode == "test":
            result["original_image1"] = image1  # 没有resize的原始大小图像
            result["original_image1"] = image2
            image1 = self.transform(image1)  # resize再标准化
            image2 = self.transform(image2)
            result["image1"] = image1
            result["image2"] = image2
            result["label"] = label // 255  # cv2读取的label，为0-255，需归一化到0-1
            label_resize = cv2.resize(label, (224, 224))  # resize label
            label_resize = np.where(label_resize != 0, 1, 0)  # 插值之后label会存在0-255之间的值
            label_resize = TF.to_tensor(label_resize)  # 转换为tensor

            result["resize_label"] = label_resize
            result["imageName"] = str(base_name)
            return result
        else:
            return