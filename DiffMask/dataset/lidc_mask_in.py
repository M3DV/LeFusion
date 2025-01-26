import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchio as tio
import os
import glob

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.Clamp(out_min=-1000, out_max=400),
    tio.RescaleIntensity(in_min_max=(-1000, 400),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(32, 64, 64))
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(32, 64, 64))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

class LIDCMASKInDataset(Dataset):
    def __init__(self, root_dir, test_txt_path, augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = test_txt_path
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS

    def get_file_names(self):
        all_file_names = glob.glob(os.path.join(self.root_dir, './**/*.nii.gz'), recursive=True)
        test_file_names = set()
        with open(self.remove_test_path, 'r') as file:
            for line in file:
                test_file_name = line.strip() 
                test_file_names.add(test_file_name)
        filtered_file_names = [
            f for f in all_file_names
            if os.path.basename(f)[:-7] not in test_file_names  
        ]
        return sorted(filtered_file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        img = tio.ScalarImage(path)
        mask_path = path.replace("/Image/", "/Mask/")
        filename = mask_path.split('/')[-1]
        new_filename = filename.replace("Vol", "Mask")
        mask_path = mask_path.replace(filename, new_filename)
        mask = tio.LabelMap(mask_path)  
        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)
        affine = img.affine
        mask = mask.data
        img = img.data
        hist = torch.histc(img[mask > 0], bins=16, min=-1, max=1) / mask.sum()
        if torch.sum(hist) == 0 or torch.isnan(hist).any():
            print(index, mask.sum(), "----", hist)
            print(img[mask > 0])
        sphere = []
        for c in range(mask.shape[0]):
            center, radius = self.min_enclosing_sphere(mask[c])
            sphere_mask = self.create_sphere_mask(mask[c].shape, center, radius)
            sphere.append(sphere_mask)
        sphere = torch.stack(sphere, dim=0)
        sphere = sphere * 2 -1
        return {
            'GT': img,
            'GT_name': filename,
            'gt_keep_mask': mask,
            'affine': affine,
            'sphere': sphere,
        }

    @staticmethod
    def min_enclosing_sphere(mask):
        indices = torch.nonzero(mask)
        if len(indices) == 0:
            return (0, 0, 0), 0
        points = indices.numpy()
        center = points.mean(axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        return center.astype(int), int(radius)

    @staticmethod
    def create_sphere_mask(shape, center, radius):
        Z, X, Y = np.mgrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = (Y - center[2]) ** 2 + (X - center[1]) ** 2 + (Z - center[0]) ** 2
        mask = dist_from_center <= radius ** 2
        return torch.tensor(mask, dtype=torch.uint8)



