echo "finetuning Seg Dpn92Unet using Gaofen datasets"
CUDA_VISIBLE_DEVICES=2 python mainK80Seg_Dpn92Unet_FineTune.py /media/tang/lzp/gaofen2

echo "finetuning Seg res34Unet using Gaofen datasets"
CUDA_VISIBLE_DEVICES=2 python mainK80Seg_res34Unet_FineTune.py /media/tang/lzp/gaofen2

echo "All models finetuned!"