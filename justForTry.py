import torch

from models.xview_first_model import models as xModel

model= xModel.SeResNext50_Unet_Loc(pretrained=None).cuda()
checkpoint_PATH=r"C:\Users\dell\Desktop\finetuneLogs\SeResNext50Unet_FineTuneepoch47.pth"
model_CKPT = torch.load(checkpoint_PATH)

model.load_state_dict(model_CKPT['state_dict'])
