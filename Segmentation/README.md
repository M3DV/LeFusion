# Segmentation

If you would like to evaluate the data on the segmentation model, you can download the segmentation code using the following command:

```
git clone https://github.com/MrGiovanni/DiffTumor/tree/main/STEP3.SegmentationModel
```

You will need to modify the appropriate command-line arguments in `STEP3.SegmentationModel\main.py` and adjust the image preprocessing steps as needed. Additionally, you should update the dataset loading process in `STEP3.SegmentationModel\main.py` to fit your own dataset.

In this codebase, there are three models available for selection: UNET, SwinUNETR, and nnUNet. Our evaluation is based on SwinUNETR and nnUNet, but you are free to choose any model that suits your segmentation needs. We did not use pre-trained weights in our evaluation; however, if you wish to use a pre-trained SwinUNETR model, you can download the corresponding pre-trained model and place the weights in `STEP3.SegmentationModel\pretrained_model`.