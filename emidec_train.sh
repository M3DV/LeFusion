dataset=emidec
diffusion_img_size=72
diffusion_depth_size=10
diffusion_num_channels=2
batch_size=20
dataset_root_dir=data/EMIDEC/Pathological
train_num_steps=50001
cond_dim=32
results_folder=LeFusion/LeFusion_Model/EMIDEC

python LeFusion/train/train.py \
    dataset=$dataset \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.diffusion_num_channels=$diffusion_num_channels \
    model.batch_size=$batch_size \
    dataset.root_dir=$dataset_root_dir \
    model.train_num_steps=$train_num_steps \
    model.cond_dim=$cond_dim \
    model.results_folder=$results_folder \
    model.batch_size=$batch_size \

