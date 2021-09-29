


echo "training Seg SeResNext50Unet using WHU datasets"
CUDA_VISIBLE_DEVICES=3 python mainK80Seg_SeResNext50Unet.py /media/tang/lzp/whu_buildings

echo "All models trained!"