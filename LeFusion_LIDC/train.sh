test_txt_dir=data/LIDC-IDRI/Pathological/test.txt
dataset_root_dir=data/LIDC-IDRI/Pathological/Image
python train/train.py dataset.test_txt_dir=$test_txt_dir dataset.root_dir=$dataset_root_dir 
