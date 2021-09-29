import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from mmcv.utils import Config
from models.ResnetCD import ResNet
import time
import sys
from tools import visualize
import imageio
from tools import visualize
# from models.model import Encoder_Decoder
from models import U_net
from tqdm import tqdm, trange
from tools import utils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import evaluation.metrics as metrics
from tensorboardX import SummaryWriter
from lib import diceloss
from tools import initialize

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
    path = os.path.join(configs.save, 'checkpoints')
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
    train_set = CDImgDataset(input_file, train_transform, label_norm=1)
    val_set = CDImgDataset(input_file, val_transform, mode="test")
    train_loader = DataLoader(train_set, batch_size=configs.data.batchsize, num_workers=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True)
    # train_loader=DataLoader(train_set,batch_size=configs.data.batchsize,shuffle=True,drop_last=True)
    loss = nn.CrossEntropyLoss()
    model = ResNet(3, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    max_epochs = 200
    writer = SummaryWriter('log_files/cd')
    for epoch in range(max_epochs):
        batch_start_time = time.time()
        train_loss = 0.0
        model.train()
        epoch_loss = 0.0
        # unloader = transforms.ToPILImage()
        if (epoch + 1) % configs.schedule.save_frequence == 0:
            save_checkpoint(epoch, model, optimizer, configs, module="cd")
        for i, data in enumerate(train_loader):
            # print(data["image"][:,1,:,:].mean(),data["image"][:,1,:,:].std())
            optimizer.zero_grad()
            train_pred = model(data["image1"].cuda(), data["image2"].cuda())
            batch_loss = loss(train_pred, torch.squeeze(data["label"].long().cuda()))
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            epoch_loss += batch_loss.item()
            if i % configs.schedule.print_frequence == 0:
                print('[%03d/%03d] [%03d/%03d] %2.2f sec(s)  Loss: %3.6f ' % (
                    epoch + 1, max_epochs, i, len(train_set) // configs.data.batchsize, time.time() - batch_start_time,
                    train_loss / len(train_set)))
                train_loss = 0.0
                batch_start_time = time.time()
        writer.add_scalar("seg epoch train loss", epoch_loss, global_step=epoch)
        if (epoch + 1) % configs.schedule.val_frequence == 0:
            model.eval()
            batch_loss = 0
            p_class, r_class, f_class = 0, 0, 0
            data_len = len(val_loader)
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    val_pred = model(data["image1"].cuda(), data["image2"].cuda())

                    batch_loss += loss(val_pred, torch.squeeze(data["resize_label"], 1).long().cuda())
                    val_pred = TF.resize(val_pred, 512)
                    om = torch.argmax(val_pred, dim=1).detach().cpu()
                    pre_label = om
                    truth = data["label"]
                    p_class_batch, r_class_batch, f_class_batch = metrics.getPrecision_Recall_F1(truth.flatten(),
                                                                                                 pre_label.flatten(),
                                                                                                 np.unique(
                                                                                                     truth.flatten()))
                    p_class += p_class_batch[-1]
                    r_class += r_class_batch[-1]
                    f_class += f_class_batch[-1]
            print("val loss : %f | average precision : %f | average recall %f | average f1 : %f" % (
                batch_loss / data_len, p_class / data_len, r_class / data_len, f_class / data_len))
            writer.add_scalar("seg epoch val loss", batch_loss / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val precision", p_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val recall", r_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val f1", f_class / data_len, global_step=epoch)
    writer.close()


def train_seg(input_file, configs):
    train_set = NormalImgDataset(input_file, train_transform, label_norm=1)
    val_set = NormalImgDataset(input_file, val_transform, mode="test")
    train_loader = DataLoader(train_set, batch_size=configs.data.batchsize, num_workers=32, shuffle=True)
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True)
    loss_weight = torch.tensor([0.5,1.2]) # 损失权重
    loss_weight = loss_weight.cuda()
    loss = nn.CrossEntropyLoss(loss_weight)
    # loss = diceloss.DiceLoss() # dice loss
    # model = Encoder_Decoder(configs).cuda() # 这个是swin-transformer
    model = U_net.R2AttU_Net(3, 2).cuda()  # 用的R2AttU_Net，比较新的U-net
    initialize.init_weights(model)  # 网络权重初始化，默认为normal
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    max_epochs = 80
    writer = SummaryWriter('log_files/seg_whu')  # log日志存储路径
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        batch_start_time = time.time()
        train_loss = 0.0
        model.train()
        # unloader = transforms.ToPILImage()
        if (epoch + 1) % configs.schedule.save_frequence == 0:
            save_checkpoint(epoch, model, optimizer, configs, module="R2AttU_Net")
        for i, data in enumerate(train_loader):
            # print(data["image"][:,1,:,:].mean(),data["image"][:,1,:,:].std())
            optimizer.zero_grad()
            train_pred = model(data["image"].cuda())
            # 这部分是dice loss的设置， dice loss标签需要做一些变换
            # dice loss 标签和预测结果shape一样
            # ____start_____
            # true_label = torch.squeeze(data["label"].long().cpu())
            # true_label = utils.get_one_hot(true_label,2)
            # true_label = true_label.permute(0,3,1,2)
            # batch_loss = loss(train_pred, true_label.cuda())
            # ______end______

            # train_pred : [b,2,h,w]
            # label : [b,h,w]
            batch_loss = loss(train_pred, torch.squeeze(data["label"].long().cuda(), dim=1))
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            epoch_loss += batch_loss.item()
            # 输出训练情况
            if i % configs.schedule.print_frequence == 0:
                print('[%03d/%03d] [%03d/%03d] %2.2f sec(s)  Loss: %3.6f ' % (
                    epoch + 1, max_epochs, i, len(train_set) // configs.data.batchsize, time.time() - batch_start_time,
                    train_loss / len(train_set)))
                train_loss = 0.0
                batch_start_time = time.time()
        writer.add_scalar("seg epoch train loss", epoch_loss, global_step=epoch)
        # 验证过程
        if (epoch + 1) % configs.schedule.val_frequence == 0:
            model.eval()  # 设置为验证状态，关闭dropout和bn
            batch_loss = 0
            p_class, r_class, f_class = 0, 0, 0
            data_len = len(val_loader)
            with torch.no_grad():  # 阻断gradient
                for i, data in enumerate(val_loader):
                    val_pred = model(data["image"].cuda())  # 【b，2，h，w】
                    batch_loss += loss(val_pred, torch.squeeze(data["resize_label"], 1).long().cuda())
                    # om = torch.argmax(val_pred.squeeze(), dim=1).detach().cpu().numpy()
                    val_pred = TF.resize(val_pred, 512)  # 将预测结果resize到512
                    om = torch.argmax(val_pred, dim=1).detach().cpu()  # 取前景背景最大值
                    pre_label = om
                    truth = data["label"]  # 标签，大小为原始大小（512*512）
                    p_class_batch, r_class_batch, f_class_batch = metrics.getPrecision_Recall_F1(truth.flatten(),
                                                                                                 pre_label.flatten(),
                                                                                                 np.unique(
                                                                                                     truth.flatten()))
                    p_class += p_class_batch[-1]  # -1指索引第二类评分，即建筑物类别评分
                    r_class += r_class_batch[-1]
                    f_class += f_class_batch[-1]
            print("val loss : %f | average precision : %f | average recall %f | average f1 : %f" % (
                batch_loss / data_len, p_class / data_len, r_class / data_len, f_class / data_len))
            writer.add_scalar("seg epoch val loss", batch_loss / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val precision", p_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val recall", r_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val f1", f_class / data_len, global_step=epoch)

    writer.close()  # 存储log


def try_cd(input_file, output_file, configs):
    """ 用于Cd debug """
    train_set = CDImgDataset(input_file, test_transform, "train", label_norm=1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    unloader = transforms.ToPILImage()
    for i, data in enumerate(train_loader):
        image1 = unloader(data["image1"][0])
        image2 = unloader(data["image2"][0])
        label = data["label"][0][0]
        figure, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(image1)
        ax[0, 1].imshow(image2)
        ax[1, 1].imshow(label)
        plt.show()


def try_seg(input_file, configs):
    """ 用于Seg debug """
    train_set = NormalImgDataset(input_file, test_transform, "test", label_norm=1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    unloader = transforms.ToPILImage()
    for i, data in enumerate(train_loader):
        image = unloader(data["image"][0])
        label = data["label"][0][0]
        figure, ax = plt.subplots(1, 2)
        '''
        ax[0].imshow(image)
        ax[1].imshow(label)
        plt.show()
        '''


if __name__ == '__main__':
    input_file = sys.argv[1]

    configs = "configs/lzp_configs.py"
    configs = Config.fromfile(configs)
    # main(input_file, output_file, configs)
    # test_for_swin_transformer(input_file, output_file, configs)
    # test_cd(input_file, output_file, configs)
    # try_seg('data/gaofen_tiny', configs)
    # train_seg(input_file,output_file,configs)
    train_seg(input_file, configs)
    # Tiaoshi(input_file, output_file, configs)
