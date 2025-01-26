data_type=emidec
types=1
diffusion_img_size=72
diffusion_depth_siz=10
diffusion_num_channels=2
batch_size=4
dataset_root_dir=data/EMIDEC/Pathological/
target_img_path=data/EMIDEC/gen/Image/
target_label_path=data/EMIDEC/gen/Mask/
model_path=LeFusion/LeFusion_Model/EMIDEC/emidec.pt
jump_length=5
jump_n_sample=5
cond_dim=32

python LeFusion/inference/inference.py \
    data_type=$data_type \
    types=$types\
    diffusion_img_size=$diffusion_img_size \
    diffusion_depth_size=$diffusion_depth_siz \
    diffusion_num_channels=$diffusion_num_channels \
    dataset_root_dir=$dataset_root_dir \
    target_img_path=$target_img_path \
    target_label_path=$target_label_path \
    schedule_jump_params.jump_length=$jump_length \
    schedule_jump_params.jump_n_sample=$jump_n_sample \
    model_path=$model_path \
    cond_dim=$cond_dim \
    batch_size=$batch_size



