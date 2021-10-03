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


def load_checkpoint(model, checkpoint_PATH):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    return model


def test_seg(input_file, output_file, configs):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
        print(output_file + "dir maked")
    # model = Encoder_Decoder(configs).cuda()
    model1 = xModel.Dpn92_Unet_Loc(pretrained=None).cuda()
    model2 = xModel.SeResNext50_Unet_Loc(pretrained=None).cuda()
    model3 = xModel.Res34_Unet_Loc(pretrained=None).cuda()
    model4 = xModel.SeNet154_Unet_Loc(pretrained=None).cuda()
    model5 = deeplab_xception.DeepLabv3_plus(pretrained=False).cuda()
    model6 = deeplab_resnet.DeepLabv3_plus(pretrained=False).cuda()

    model1 = load_checkpoint(model1, configs["pre_train_seg_model1"])
    model2 = load_checkpoint(model2, configs["pre_train_seg_model2"])
    model3 = load_checkpoint(model3, configs["pre_train_seg_model3"])
    model4 = load_checkpoint(model4, configs["pre_train_seg_model4"])
    model5 = load_checkpoint(model5, configs["pre_train_seg_model5"])
    model6 = load_checkpoint(model6, configs["pre_train_seg_model6"])

    model1 = model1.cuda()
    model2 = model2.cuda()
    model3 = model3.cuda()
    model4 = model4.cuda()
    model5 = model5.cuda()
    model6 = model6.cuda()

    test_set = GaofenDataset.NormalImgDataset(input_file, train_transform, mode="test")
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=True)

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()

    p_class, r_class, f_class = 0, 0, 0
    data_len = len(test_loader)
    # unloader = transforms.ToPILImage()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            val_pred1 = model1(data["image"].cuda())
            val_pred2 = model2(data["image"].cuda())
            val_pred3 = model3(data["image"].cuda())
            val_pred4 = model4(data["image"].cuda())
            val_pred5 = model5(data["image"].cuda())
            val_pred6 = model6(data["image"].cuda())

            val_pred1 = torch.sigmoid(val_pred1)
            val_pred2 = torch.sigmoid(val_pred2)
            val_pred3 = torch.sigmoid(val_pred3)
            val_pred4 = torch.sigmoid(val_pred4)
            val_pred5 = torch.sigmoid(val_pred5)
            val_pred6 = torch.sigmoid(val_pred6)

            val_pred = torch.cat([val_pred1, val_pred2, val_pred3, val_pred4,val_pred5,val_pred6], dim=1)
            val_pred = torch.mean(val_pred, dim=1)
            # val_pred = torch.max(val_pred,dim=1)
            val_pred = val_pred.detach().cpu().numpy()
            #val_pred = val_pred.values.clone().detach()
            #val_pred = val_pred.cpu().numpy()
            pre_mask_rgb = cv2.resize(val_pred[0], (512, 512))
            pre_mask_rgb = np.where(pre_mask_rgb > 0.35, 1, 0)  # 0.15
            pre_label = pre_mask_rgb
            truth = data["label"]
            p_class_batch, r_class_batch, f_class_batch = metrics.getPrecision_Recall_F1(truth.flatten(),
                                                                                         pre_label.flatten(),
                                                                                         np.unique(truth.flatten()))
            p_class += p_class_batch[-1]  # -1指索引第二类评分，即建筑物类别评分
            r_class += r_class_batch[-1]
            f_class += f_class_batch[-1]
        print(" average precision : %f | average recall %f | average f1 : %f" % (
            p_class / data_len, r_class / data_len, f_class / data_len))


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    configs = "configs/lzp_configs_gaofen_ensemble"
    configs = loadConfigs.readConfigs(configs)
    # test_cd(input_file,output_file,configs)
    test_seg(input_file, output_file, configs)
