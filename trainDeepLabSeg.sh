echo "training Seg res101DeepLabV3P using WHU datasets"
CUDA_VISIBLE_DEVICES=4 python mainK80Seg_res101DeepLabV3P.py /media/tang/lzp/whu_buildings

echo "training Seg xceptionDeepLabV3P using WHU datasets"
CUDA_VISIBLE_DEVICES=4 python mainK80Seg_xceptionDeepLabV3P.py /media/tang/lzp/whu_buildings

echo "All models trained!"