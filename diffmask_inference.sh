name=lidc_mask
dataset_root_dir=data/LIDC/Pathological/Image
test_txt_path=data/LIDC/Pathological/test.txt
gen_mask_path=data/GenMask/LIDC
diffusion_img_size=64
diffusion_depth_size=32
out_dim=1
unet_num_channels=2
model_path=DiffMask/DiffMask_Model/diffmask.pt

python DiffMask/inference/inference.py \
    name=$name \
    dataset_root_dir=$dataset_root_dir \
    test_txt_path=$test_txt_path \
    gen_mask_path=$gen_mask_path \
    diffusion_img_size=$diffusion_img_size \
    diffusion_depth_size=$diffusion_depth_size \
    out_dim=$out_dim \
    unet_num_channels=$unet_num_channels \
    out_dim=$out_dim \
    model_path=$model_path \


