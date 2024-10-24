# LeFusion

LeFusion: Lesion-Focused Diffusion Model. The top illustrates the training process of LeFusion, while the bottom shows the inference. During training, LeFusion avoids learning unnecessary background generation using a lesion-focused loss. In inference, by combining forward-diffused real backgrounds with reverse-diffused generated foregrounds, LeFusion ensures high-quality background generation. Additionally, we introduce histogram-based texture control to handle multi-peak lesions and multi-channel decomposition for multi-class lesions. ([arXiv](https://arxiv.org/abs/2403.14066))

![lefusion_model](https://github.com/M3DV/LeFusion/blob/main/figures/lefusion_model.png)

## :bookmark_tabs:Data Preparation

We utilized the LIDC dataset ([PubMed](https://pubmed.ncbi.nlm.nih.gov/21452728/#:~:text=Methods)), which includes 1,010 chest CT scans. From these, we extracted 2,624 pathology regions of interest (ROIs) related to lung nodules to train the LeFusion Model. The dataset is divided into 808 cases for training, containing 2,104 lung nodule ROIs, and 202 cases for testing, containing 520 lung nodule ROIs. This portion of the dataset is located in `LIDC-IDRI\Pathological`, with the `test.txt` listing the data used for testing.

Additionally, we provide 20 normal ROIs from healthy patients, representing areas where lung nodules typically appear. This data is located in `LIDC-IDRI\Normal`, where `Image` contains the healthy images, and `Mask` includes the corresponding masks generated by matching lung and ground truth masks, which can be used to generate lesions. You can simulate lesion generation on the Normal dataset.

Furthermore, we provide pre-generated images with lesions based on the `LIDC-IDRI\Normal` dataset. These images are stored in `LIDC-IDRI\Demo`, where `Image_i` represents the images generated under the control information *hist_i*. The pre-trained weights used to generate these images are available in the pre-trained weights mentioned below ([HuggingFace🤗](https://huggingface.co/YuheLiuu/LeFusion/tree/main/LIDC_LeFusion_Model)).

```
├── LIDC-IDRI
    ├── Pathological
    │   ├── Image
    │   ├── Mask
    │   └── test.txt
    ├── Normal
    │   ├── Image
    │   └── Mask
    └── Demo
        ├── Image
        │   ├── Image_1
        │   ├── Image_2
        │   └── Image_3
        └── Mask
            ├── Mask_1
            ├── Mask_2
            └── Mask_3
```

## :nut_and_bolt: Installation

1. Create a virtual environment `conda create -n lefusion python=3.10` and activate it `conda activate lefusion`
2. Download the code`git clone https://github.com/M3DV/LeFusion.git`
3. Check if your pip version is 22.3.1. If it is not, install pip version 22.3.1 `pip install pip==22.3.1`
4. Enter the LeFusion folder `cd LeFusion/LeFusion_LIDC` and run `pip install -r requirements.txt`

## :bulb:Get Started

1. Download the LIDC_IDRI dataset ([HuggingFace🤗](https://huggingface.co/datasets/YuheLiuu/LIDC-IDRI/tree/main))

   In our study, the LeFusion Model focuses on the generation of lung nodule regions.If you want to train a Diffusion Model to synthesize lung nodules, you can use the LIDC-IDRI dataset that has already been processed by us to train the LeFusion Model. Just put the LIDC-IDRI dataset to `LeFusion/LeFusion_LIDC/data`.

   > ✨**Note**: Before running the following command, make sure you are inside the `LeFusion/LeFusion_LIDC` folder. 

   ```bash
   mkdir data
   cd data
   mkdir LIDC-IDRI
   cd LIDC-IDRI
   wget https://huggingface.co/datasets/YuheLiuu/LIDC-IDRI/resolve/main/Pathological.tar -O Pathological.tar
   tar -xvf Pathological.tar
   wget https://huggingface.co/datasets/YuheLiuu/LIDC-IDRI/resolve/main/Normal.tar -O Normal.tar
   tar -xvf Normal.tar
   wget https://huggingface.co/datasets/YuheLiuu/LIDC-IDRI/resolve/main/Demo.tar -O Demo.tar
   tar -xvf Demo.tar
   
   ```

2. Download the pre-trained LeFusion Model ([HuggingFace🤗](https://huggingface.co/YuheLiuu/LeFusion/tree/main/LIDC_LeFusion_Model))

   We offer the pre-trained  LeFusion Model, which has been trained for 50,001 steps on the LIDC-IDRI dataset. This pre-trained model can be directly used for Inference if you do not want to re-train the LeFusion Model. Simply download it to `LeFusion/LeFusion_model`.

   ```bash
   cd ..
   mkdir LeFusion_model
   cd LeFusion_model
   wget https://huggingface.co/YuheLiuu/LeFusion/resolve/main/LIDC_LeFusion_Model/model-50.pt -O model-50.pt
   ```

   If you have downloaded the pre-trained model, you can skip the training step and proceed directly to inference!

## :microscope:Train LeFusion Model

Start training:

> ✨**Note**: Before running the following command, make sure you are inside the `LeFusion/LeFusion_LIDC` folder.

```bash
test_txt_dir=data/LIDC-IDRI/Pathological/test.txt
dataset_root_dir=data/LIDC-IDRI/Pathological/Image
train_num_steps=50001
python train/train.py dataset.test_txt_dir=$test_txt_dir dataset.root_dir=$dataset_root_dir model.train_num_steps=$train_num_steps
```

Notably, `data_path` actually refers to the directory location of the corresponding images. Additionally, the corresponding label directory should be placed in the same folder as the image directory and should be named `Mask`.

Our model was trained for 50,000 steps using five 40GB A100 GPUs, taking two and a half days. However, we found that the model performs very well after 20,000 steps. Therefore, when training a model on your own, anywhere between 20,000 to 50,000 steps would yield good results. Additionally, by default, we save the weights every 1,000 steps, and you can modify the relevant parameters in `LeFusion/train/config`.

## :chart_with_upwards_trend:Inference

Start inference:

> ✨**Note**: Before running the following command, make sure you are inside the `LeFusion/LeFusion_LIDC` folder.

```bash
test_txt_dir=data/LIDC-IDRI/Pathological/test.txt
dataset_root_dir=data/LIDC-IDRI/Normal/Image/
target_img_path=data/LIDC-IDRI/gen/Image/
target_label_path=data/LIDC-IDRI/gen/Mask/
jump_length=5
jump_n_sample=5
python test/inference.py test_txt_dir=$test_txt_dir dataset_root_dir=$dataset_root_dir target_img_path=$target_img_path target_label_path=$target_label_path schedule_jump_params.jump_length=$jump_length schedule_jump_params.jump_n_sample=$jump_n_sample
```

Three folders, Image_1, Image_2, and Image_3, will be generated under the` target_img_path` directory, each representing images generated under the control of hist_1, hist_2, and hist_3 respectively. Similarly, three folders will be generated under the Mask directory, but unlike the Image folders, files with the same name in each of the three Mask folders contain the same mask.

For *jump_length* and *jump_n_sample*, larger values generally result in longer image generation times. We found that when these two parameters are between 2 and 10, the generated images maintain good quality. When both parameters are set to 2, it takes about 40 seconds to generate an image using a 40G A100 GPU.

## :mag_right:Visualization

![visualization](https://github.com/M3DV/LeFusion/blob/main/figures/visualization.jpg)
The first image is a healthy image from `LIDC-IDRI/Normal`. The second image is the corresponding generated mask, where lesions will be generated in the areas marked by the mask. Image_1, Image_2, and Image_3 are the lesions generated when the control information is set to Hist_1, Hist_2, and Hist_3, respectively.

## Citation

```
@misc{zhang2024lefusioncontrollablepathologysynthesis,
      title={LeFusion: Controllable Pathology Synthesis via Lesion-Focused Diffusion Models}, 
      author={Hantao Zhang and Yuhe Liu and Jiancheng Yang and Shouhong Wan and Xinyuan Wang and Wei Peng and Pascal Fua},
      year={2024},
      eprint={2403.14066},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2403.14066}, 
}
```

## Acknowledgement

Some of our code is modified based on [medicaldiffusion](https://github.com/FirasGit/medicaldiffusion) and [RePaint](https://github.com/andreas128/RePaint), and we greatly appreciate the efforts of the respective authors for providing open-source code. We also thank [DiffTumor](https://github.com/MrGiovanni/DiffTumor/tree/main/STEP3.SegmentationModel) for providing the segmentation model code.

## ToDo List

✅ **The preprocessed LIDC-IDRI dataset**  🚀

✅ **The LeFusion model applied to LIDC-IDRI** 🚀

🔲 **The DiffMask model used for generating mask** 

🔲 **The preprocessed EMIDC dataset**  

🔲 **The LeFusion model applied to EMIDC**  
