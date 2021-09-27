data = dict(batchsize=8)
schedule = dict(print_frequence=20,save_frequence=10,val_frequence=1)
decoder = dict(ppm_in_dim=768,each_out_dim=[96,192,384,768],reduction_dim=96)
save = "save_file_whu"
# pre_train_swin_transformer_model = "save_file/checkpoints/swintransformer_epoch199.pth"
pre_train_swin_transformer_model = "/home/tang/桌面/liu/MyCDCode2/MyCDCode/save_file_whu/checkpoints/R2AttU_Netepoch9.pth"
pre_train_cd_model = "save_file/checkpoints/cd_epoch199.pth"
