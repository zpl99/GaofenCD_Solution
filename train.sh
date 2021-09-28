echo "training Seg Dpn92Unet using WHU datasets"
CUDA_VISIBLE_DEVICES=3 python mainK80Seg_Dpn92Unet.py /media/tang/lzp/whu_buildings

echo "training Seg res34Unet using WHU datasets"
CUDA_VISIBLE_DEVICES=3 python mainK80Seg_res34Unet.py /media/tang/lzp/whu_buildings

echo "training Seg SeNet154Unet using WHU datasets"
CUDA_VISIBLE_DEVICES=3 python mainK80Seg_SeNet154Unet.py /media/tang/lzp/whu_buildings

echo "training Seg SeResNext50Unet using WHU datasets"
CUDA_VISIBLE_DEVICES=3 python mainK80Seg_SeResNext50Unet.py /media/tang/lzp/whu_buildings

echo "All models trained!"