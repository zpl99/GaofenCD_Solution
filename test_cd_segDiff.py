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
from tqdm import tqdm, trange
from tools import utils, loadConfigs
import torchvision.transforms.functional as TF
from models import U_net
from dataset import WHUDataset, GaofenDataset, CDdataset0_5
from PIL import Image

# encoding=utf8
"""
高分比赛代码1.0
run.py 只保留了测试部分
checkpoints路径可以通过lzp_configs.py设定
前后时相语义分割采用swintransformer+UperNet
前后时相变化检测采用两个Resnet-50
训练时单独训练，测试时两个模型各自输出结果，相互之间没关系
训练数据集组织
"""
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.33, 0.34, 0.30],
                         std=[0.17, 0.165, 0.17])
])
label_transform = transforms.Compose([
    transforms.Resize(224)
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
""" 比赛版本！python run.py inputfiles outputfiles"""


def save_checkpoint(epoch, model, optimizer, configs):
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    path = os.path.join(configs.save, 'checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + "dir maked")
    save_name = os.path.join(path, 'epoch%d.pth' % epoch)
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
        self.x = self.parseInput(input_file)
        self.transform = transform
        self.image_files = os.listdir(self.x)

    def parseInput(self, input_file):
        # TODO： 图像的输入路径，需要调整成为比赛规范格式
        x = os.path.join(input_file)
        return x

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        result = {}
        image_path = self.image_files[index]
        base_name = os.path.basename(image_path)
        # print(base_name)
        image = cv2.imread(os.path.join(self.x, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        result["image"] = image
        base_name_list = base_name.split(".")
        base_name_no_png = base_name_list[0]
        result["imageName"] = base_name_no_png  # 用于输出二值图的文件命名
        return result


class CDImgDataset(Dataset):
    def __init__(self, input_file, transform=None, mode="train", label_norm=1):
        """label_norm : whu_building 255"""
        self.label_norm = label_norm
        self.mode = mode
        self.x, self.imageName = self.parseInput(input_file)
        self.transform = transform
        self.image_files = os.listdir(self.x)

    def parseInput(self, input_file):
        x = os.path.join(input_file)
        imageNames = utils.readImageName(input_file)  # 适合于gaofen数据的命名格式
        return x, imageNames

    def __len__(self):
        return len(self.imageName)

    def __getitem__(self, index):
        result = {}
        base_name = int(self.imageName[index])

        # print(base_name)
        image1 = cv2.imread(os.path.join(self.x, str(base_name) + "_1.png"))

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread(os.path.join(self.x, str(base_name) + "_2.png"))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        result["original_image1"] = image1
        result["original_image1"] = image2
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        result["image1"] = image1
        result["image2"] = image2
        result["imageName"] = str(base_name)
        return result


def test_cd(input_file, output_file, configs):
    """改成了Seg做差"""
    if not os.path.exists(output_file):
        os.makedirs(output_file)
        print(output_file + "dir maked")
    model = U_net.R2AttU_Net(3,1).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = load_checkpoint(model, configs["pre_train_seg_model"], optimizer)
    model = model.cuda()
    test_set = CDImgDataset(input_file, train_transform, mode="test")
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=True)
    model.eval()
    # unloader = transforms.ToPILImage()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            val_pred_1 = model(data["image1"].cuda())
            val_pred_2 = model(data["image2"].cuda())
            val_pred_1 = torch.sigmoid(val_pred_1)
            val_pred_2 = torch.sigmoid(val_pred_2)

            val_pred_1 = val_pred_1.detach().cpu().numpy() # tensor to numpy
            val_pred_2 = val_pred_2.detach().cpu().numpy()
            pre_mask_1 = cv2.resize(val_pred_1[0][0], (512, 512)) # 224 to 512
            pre_mask_2 = cv2.resize(val_pred_2[0][0], (512, 512)) # 224 to 512
            pre_mask_1 = np.where(pre_mask_1 > 0.10, 255, 0) # 阈值设定为0.3
            pre_mask_2 = np.where(pre_mask_2 > 0.10, 255, 0)  # 阈值设定为0.3
            pre_label = np.abs(pre_mask_1 - pre_mask_2)
            pre_mask_PIL = Image.fromarray(np.uint8(pre_label))  # 转成PIL Image
            pre_mask_single = TF.to_grayscale(pre_mask_PIL, 1)  # 转成灰度图
            output_path = os.path.join(output_file, "%s_change.png" % data["imageName"][0])  # 存储路径
            imageio.imwrite(output_path, pre_mask_single)  # 写入图像


def test_seg(input_file, output_file, configs):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
        print(output_file + "dir maked")
    # model = Encoder_Decoder(configs).cuda()
    model = U_net.R2AttU_Net(3, 1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model, optimizer = load_checkpoint(model, configs["pre_train_seg_model"], optimizer)
    model = model.cuda()
    test_set = NormalImgDataset(input_file, train_transform, mode="test")
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=True)
    model.eval()
    # unloader = transforms.ToPILImage()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            val_pred = model(data["image"].cuda())
            val_pred = torch.sigmoid(val_pred)
            val_pred = val_pred.detach().cpu().numpy()
            pre_mask_rgb = cv2.resize(val_pred[0][0], (512, 512))
            pre_mask_rgb = np.where(pre_mask_rgb > 0.10, 255, 0)
            pre_mask_PIL = Image.fromarray(np.uint8(pre_mask_rgb))
            # TODO：需要check一下pre_mask_single里面是否有非0-255的值
            pre_mask_single = TF.to_grayscale(pre_mask_PIL, 1)
            output_path = os.path.join(output_file, "%s_label.png" % data["imageName"][0])
            imageio.imwrite(output_path, pre_mask_single)


def mainFunction(input_file, output_file, configs):
    pass
    # test_for_swin_transformer(input_file, output_file, configs)
    # print("binary mask pre finished!")
    # test_cd(input_file, output_file, configs)
    # print("cd mask pre finished!")


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    configs = "configs/lzp_configs_gaofen"
    configs = loadConfigs.readConfigs(configs)
    # configs = Config.fromfile(configs)
    test_cd(input_file, output_file, configs)
    test_seg(input_file, output_file, configs)

