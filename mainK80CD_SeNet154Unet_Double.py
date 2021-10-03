import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
# from mmcv.utils import Config
from models.ResnetCD import ResNet
import time
import sys
from tools import visualize
import imageio
from tools import visualize
# from models.model import Encoder_Decoder
from models import U_net
from tqdm import tqdm, trange
from tools import utils, loadConfigs
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import evaluation.metrics as metrics
from tensorboardX import SummaryWriter
from lib import focalLoss
from tools import initialize, getBestCheckPoints
from dataset import CDdataset0_5, WHUDataset, GaofenDataset
from models.xview_first_model import models as xModel
from models.deeplabV3Plus import deeplab_resnet, deeplab_xception
import gc

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.33, 0.34, 0.30],
                         std=[0.17, 0.165, 0.17])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.33, 0.34, 0.30],
                         std=[0.17, 0.165, 0.17])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),

])

"""
训练数据集组织
GaofenData/
        images/
                train/
                            1_1.png
                            ......
        gt/
                train/
                            1_1_label.png
                            1_2_label.png
                            1_change.png
                            ......

python main.py input_file_path
configs目录下可以设置部分参数
"""


def save_checkpoint(epoch, model, optimizer, configs, module="swintransformer"):
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    path = os.path.join(configs["save_model_path"], 'checkpoints')
    print("saving........")
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + "dir maked")
    save_name = os.path.join(path, module + 'epoch%d.pth' % epoch)
    torch.save(save_state, save_name)
    print('Saved model')


def load_checkpoint(model, checkpoint_PATH, optimizer):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer


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
        # label = cv2.imread(os.path.join(self.y, label_path + "_label.png"), cv2.IMREAD_GRAYSCALE)
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
        if self.mode == "compete":
            imageNames = utils.readImageName(input_file)  # 适合于比赛模式 ，暂时用不到
        else:
            imageNames = utils.readImageName(x)  # 用来读取图像名

        return x, y, imageNames

    def __len__(self):
        return len(self.imageName)

    def __getitem__(self, index):
        result = {}
        base_name = int(self.imageName[index])

        # print(base_name)
        image1 = cv2.imread(os.path.join(self.x, str(base_name) + "_1.png"))  # imagename_1.png
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # cv2 读取下来是BGR，所以需要强制更改通道顺序为RGB
        image2 = cv2.imread(os.path.join(self.x, str(base_name) + "_2.png"))  # imagename_2.png
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        label_path = base_name
        label = cv2.imread(os.path.join(self.y, str(label_path) + "_change.png"), cv2.IMREAD_GRAYSCALE)  # 【h，w】
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


def train_cd(input_file, configs):
    train_set = GaofenDataset.CDImgDataset(input_file, train_transform, label_norm=1)
    val_set = GaofenDataset.CDImgDataset(input_file, val_transform, mode="test")
    train_loader = DataLoader(train_set, batch_size=configs["data"]["batchsize"], num_workers=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=configs["data"]["val_batchsize"], num_workers=16, shuffle=True)
    loss = nn.BCEWithLogitsLoss()
    model = xModel.SeNet154_Unet_Double(pretrained=None).cuda()
    initialize.init_weights(model, init_type='kaiming')
    checkpoint = torch.load(configs["pre_train_seg"])
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    # 垃圾回收
    del loaded_dict
    del sd
    del checkpoint
    gc.collect()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015)
    torch.cuda.empty_cache()
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.000001, last_epoch=-1)
    max_epochs = 100
    writer = SummaryWriter(os.path.join("log_files", configs["log_save_path"]))  # TODO:根据需求修改
    saver = getBestCheckPoints.CheckPointsSaver(configs["save_model_name"])
    model = nn.DataParallel(model).cuda()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        batch_start_time = time.time()
        train_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data["image1"].cuda(), data["image2"].cuda())
            batch_loss = loss(train_pred, data["label"].float().cuda())
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            epoch_loss += batch_loss.item()
            # 输出训练情况
            if i % configs["schedule"]["print_frequence"] == 0:
                print('[%03d/%03d] [%03d/%03d] %2.2f sec(s)  Loss: %3.6f ' % (
                    epoch + 1, max_epochs, i, len(train_set) // configs["data"]["batchsize"],
                    time.time() - batch_start_time,
                    train_loss / configs["schedule"]["print_frequence"]))
                train_loss = 0.0
                batch_start_time = time.time()
        writer.add_scalar("seg epoch train loss", (epoch_loss * configs["data"]["batchsize"]) / len(train_loader),
                          global_step=epoch)
        lr_schedule.step()  # 更新学习率
        lr_new = lr_schedule.get_lr()
        # 验证过程
        if (epoch + 1) % configs["schedule"]["val_frequence"] == 0:
            model.eval()  # 设置为验证状态，关闭dropout和bn
            batch_loss = 0
            p_class, r_class, f_class = 0, 0, 0
            data_len = len(val_loader)
            with torch.no_grad():  # 阻断gradient
                for i, data in enumerate(val_loader):
                    val_pred = model(data["image1"].cuda(), data["image2"].cuda())  # 【b，1，h，w】
                    batch_loss += loss(val_pred, data["resize_label"].float().cuda())
                    val_pred = torch.sigmoid(val_pred)
                    val_pred = val_pred.cpu().numpy()
                    om = np.where(val_pred > 0.15, 1, 0)
                    pre_label = om
                    truth = data["resize_label"]
                    p_class_batch, r_class_batch, f_class_batch = metrics.getPrecision_Recall_F1(truth.flatten(),
                                                                                                 pre_label.flatten(),
                                                                                                 np.unique(
                                                                                                     truth.flatten()))
                    p_class += p_class_batch[-1]  # -1指索引第二类评分，即建筑物类别评分
                    r_class += r_class_batch[-1]
                    f_class += f_class_batch[-1]
            # 输出验证结果并写入writer
            print(
                "val loss : %f | average precision : %f | average recall %f | average f1 : %f | learning rate : %f" % (
                    (batch_loss / data_len) * configs["data"]["val_batchsize"], p_class / data_len, r_class / data_len,
                    f_class / data_len, lr_new[0]))
            writer.add_scalar("seg epoch val loss", (batch_loss / data_len) * configs["data"]["val_batchsize"],
                              global_step=epoch)
            writer.add_scalar("seg epoch val precision", p_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val recall", r_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val f1", f_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch learning rate", lr_new[0], global_step=epoch)
            # save
            saver.push(epoch, model, optimizer, configs,
                       float(((batch_loss / data_len) * configs["data"]["val_batchsize"]).cpu()))
    writer.close()  # 存储log


def try_cd(input_file, configs):
    """ 用于Cd debug """
    train_set = CDdataset0_5.CDImgDataset(input_file, test_transform, "test", label_norm=1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    unloader = transforms.ToPILImage()
    for i, data in enumerate(train_loader):
        image1 = unloader(data["image1"][0])
        image2 = unloader(data["image2"][0])
        label = data["resize_label"][0][0]
        print(np.unique(label))
        # figure, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(image1)
        # ax[0, 1].imshow(image2)
        # ax[1, 1].imshow(label)
        # plt.show()


def try_seg(input_file, configs):
    """ 用于Seg debug """
    train_set = GaofenDataset.FenCengImgDataset(input_file, test_transform, label_norm=1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    unloader = transforms.ToPILImage()

    for i, data in enumerate(train_loader):
        image = unloader(data["image"][0])
        label = data["label"].numpy()

        figure, ax = plt.subplots(1, 2)

        ax[0].imshow(image)
        ax[1].imshow(label[0][0])
        plt.show()


if __name__ == '__main__':
    input_file = sys.argv[1]

    configs = "configs/lzp_configs_cd_SeNet154Unet_Double"
    # configs = Config.fromfile(configs) 
    configs = loadConfigs.readConfigs(configs)
    # try_seg(input_file, configs)
    # main(input_file, output_file, configs)
    # test_for_swin_transformer(input_file, output_file, configs)
    # test_cd(input_file, output_file, configs)
    # try_cd(r"new_cd_data/", configs)
    train_cd(input_file, configs)
    # Tiaoshi(input_file, output_file, configs)
