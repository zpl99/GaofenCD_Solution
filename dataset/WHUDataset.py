from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from tools import utils
import random


# WHU 语义分割数据集，路径在K80 /media/tang/lzp/whu_buildings
class NormalImgDataset(Dataset):
    def __init__(self, input_file, transform=None, mode="train", label_norm=1):
        """label_norm : whu_building 255"""
        self.label_norm = label_norm
        self.mode = mode
        self.x, self.y = self.parseInput(input_file)
        self.transform = transform
        self.image_files = os.listdir(self.x)
        self.label_files = os.listdir(self.y)

    def parseInput(self, input_file):
        x = os.path.join(input_file, "images", self.mode)
        y = os.path.join(input_file, "gt", self.mode)
        return x, y

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        result = {}
        image_path = self.image_files[index]
        base_name = os.path.basename(image_path)
        # print(base_name)
        image = cv2.imread(os.path.join(self.x, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = base_name.split(".")
        label_path = label_path[0]

        label = cv2.imread(os.path.join(self.y, label_path + ".png"), cv2.IMREAD_GRAYSCALE)
        label = label // self.label_norm
        # print(label.shape)
        # label = label*255

        if self.mode == "train":
            image, label = utils.data_transfomr_pipline(image, label, pipline=["roate", "vflipAndhflip", "pad"],
                                                        size=224)
            label = np.where(label != 0, 1, 0)  # 插值之后label会存在0-255之间的值
            result["image"] = image
            result["label"] = label
            # print(np.unique(label))
            # print(torch.max(label))
            return result
        elif self.mode == "test":
            result["original_image"] = image
            image = self.transform(image)  # val_transform
            result["image"] = image
            result["label"] = label // 255  # label is 0-255

            base_name_list = base_name.split(".")
            base_name_no_png = base_name_list[0]
            result["imageName"] = base_name_no_png  # 用于输出二值图的文件命名
            label_resize = cv2.resize(label, (224, 224))
            label_resize = np.where(label_resize != 0, 1, 0)
            label_resize = TF.to_tensor(label_resize)
            result["resize_label"] = label_resize
            return result
        else:
            return


class FenCengImgDataset(Dataset):
    def __init__(self, input_file, transform=None, pos=0.8, label_norm=1):
        """label_norm : whu_building 255"""
        """分层抽样，只有train模式，测试模式请用NormalImageDataset"""
        self.pos = pos
        self.label_norm = label_norm
        self.x_less, self.y_less, self.x_more, self.y_more = self.parseInput(input_file)
        self.transform = transform
        self.image_files_less = os.listdir(self.x_less)
        self.image_files_more = os.listdir(self.x_more)

        self.label_files_less = os.listdir(self.y_less)
        self.label_files_more = os.listdir(self.y_more)

        self.image_files = self.image_files_less + self.image_files_more

    def parseInput(self, input_file):
        x_less = os.path.join(input_file + "/less", "images", "train")
        y_less = os.path.join(input_file + "/less", "gt", "train")

        x_more = os.path.join(input_file + "/more", "images", "train")
        y_more = os.path.join(input_file + "/more", "gt", "train")

        return x_less, y_less, x_more, y_more

    def __len__(self):
        return len(self.image_files_more)

    def __getitem__(self, index):
        result = {}
        random_number = random.random()
        if random_number > self.pos:
            if index>=len(self.image_files_less)-1:
                index = random.randint(0,len(self.image_files_less)-1)
            image_path = self.image_files_less[index]
            base_name = os.path.basename(image_path)
            image = cv2.imread(os.path.join(self.x_less, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_path = base_name.split(".")
            label_path = label_path[0]

            label = cv2.imread(os.path.join(self.y_less, label_path + ".png"), cv2.IMREAD_GRAYSCALE)
            label = label // self.label_norm
        else:
            image_path = self.image_files_more[index]
            base_name = os.path.basename(image_path)
            image = cv2.imread(os.path.join(self.x_more, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_path = base_name.split(".")
            label_path = label_path[0]
            label = cv2.imread(os.path.join(self.y_more, label_path + ".png"), cv2.IMREAD_GRAYSCALE)
            label = label // self.label_norm

        image, label = utils.data_transfomr_pipline(image, label, pipline=["roate", "vflipAndhflip", "pad"],
                                                    size=224)
        label = np.where(label != 0, 1, 0)  # 插值之后label会存在0-255之间的值
        result["image"] = image
        result["label"] = label

        return result
