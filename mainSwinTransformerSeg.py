import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from models.ResnetCD import ResNet
import time
import sys
from models.model import Encoder_Decoder
from models import U_net
from tqdm import tqdm, trange
from tools import utils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import evaluation.metrics as metrics
from tensorboardX import SummaryWriter
from lib import diceloss
from tools import initialize
from dataset import WHUDataset
from tools import loadConfigs

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

def save_checkpoint(epoch, model, optimizer, configs, module="swintransformer"):
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    path = os.path.join(configs["save"], 'checkpoints')
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


def train_seg(input_file, configs):
    train_set = WHUDataset.NormalImgDataset(input_file, train_transform, label_norm=1)
    val_set = WHUDataset.NormalImgDataset(input_file, val_transform, mode="test")
    train_loader = DataLoader(train_set, batch_size=configs["data"]["batchsize"], num_workers=32, shuffle=True)
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=configs["data"]["val_batchsize"], shuffle=True)
    # loss_weight = torch.tensor([0.5,1.2]) # 损失权重
    # loss_weight = loss_weight.cuda()
    loss = nn.BCEWithLogitsLoss()
    # loss = focalLoss.FocalLoss(alpha=0.75, gamma=1, logits=True, reduce=True)  # focal loss
    # loss = diceloss.DiceLoss() # dice loss
    model = Encoder_Decoder(configs).cuda() # 这个是swin-transformer
    # model = U_net.R2AttU_Net(3, 1).cuda()  # 用的R2AttU_Net，比较新的U-net
    initialize.init_weights(model)  # 网络权重初始化，默认为normal
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    max_epochs = 200
    writer = SummaryWriter('log_files/seg_whu_swin_transformer_celoss')  # log日志存储路径
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        batch_start_time = time.time()
        train_loss = 0.0
        model.train()
        # unloader = transforms.ToPILImage()
        if (epoch + 1) % configs["schedule"]["save_frequence"] == 0:
            save_checkpoint(epoch, model, optimizer, configs, module="swin_transformer_celoss")
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

            # train_pred : [b,1,h,w]
            # label : [b,1,h,w]
            # batch_loss = loss(train_pred, torch.squeeze(data["label"].long().cuda(), dim=1))
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
        writer.add_scalar("seg epoch train loss", (epoch_loss*configs["data"]["batchsize"])/len(train_loader), global_step=epoch)
        # 验证过程
        if (epoch + 1) % configs["schedule"]["val_frequence"] == 0:
            model.eval()  # 设置为验证状态，关闭dropout和bn
            batch_loss = 0
            p_class, r_class, f_class = 0, 0, 0
            data_len = len(val_loader)
            with torch.no_grad():  # 阻断gradient
                for i, data in enumerate(val_loader):
                    val_pred = model(data["image"].cuda())  # 【b，1，h，w】
                    batch_loss += loss(val_pred, data["resize_label"].float().cuda())
                    # om = torch.argmax(val_pred.squeeze(), dim=1).detach().cpu().numpy()
                    # val_pred = TF.resize(val_pred, 512)  # 将预测结果resize到512

                    val_pred = torch.sigmoid(val_pred)
                    val_pred = val_pred.cpu().numpy()
                    om = np.where(val_pred > 0.3, 1, 0)
                    pre_label = om
                    truth = data["resize_label"]
                    p_class_batch, r_class_batch, f_class_batch = metrics.getPrecision_Recall_F1(truth.flatten(),
                                                                                                 pre_label.flatten(),
                                                                                                 np.unique(
                                                                                                     truth.flatten()))
                    p_class += p_class_batch[-1]  # -1指索引第二类评分，即建筑物类别评分
                    r_class += r_class_batch[-1]
                    f_class += f_class_batch[-1]
            print("val loss : %f | average precision : %f | average recall %f | average f1 : %f" % (
                (batch_loss / data_len)*configs["data"]["val_batchsize"], p_class / data_len, r_class / data_len, f_class / data_len))
            writer.add_scalar("seg epoch val loss", (batch_loss / data_len)*configs["data"]["val_batchsize"], global_step=epoch)
            writer.add_scalar("seg epoch val precision", p_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val recall", r_class / data_len, global_step=epoch)
            writer.add_scalar("seg epoch val f1", f_class / data_len, global_step=epoch)

    writer.close()  # 存储log

if __name__ == '__main__':
    input_file = sys.argv[1]

    configs = "configs/lzp_configs_swin_transformer"
    # configs = Config.fromfile(configs)
    configs = loadConfigs.readConfigs(configs)
    # main(input_file, output_file, configs)
    # test_for_swin_transformer(input_file, output_file, configs)
    # test_cd(input_file, output_file, configs)
    # try_cd(r"new_cd_data/", configs)
    train_seg(input_file, configs)
    # Tiaoshi(input_file, output_file, configs)