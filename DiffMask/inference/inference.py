import os
import torch as th
import hydra
import sys
from omegaconf import DictConfig
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, Trainer, GaussianDiffusion_Nolatent
from get_dataset.get_dataset import get_inference_dataloader
import torchio as tio
import numpy as np
import yaml
import io
import blobfile as bf
from scipy.ndimage import binary_fill_holes, gaussian_filter, label, zoom
import numpy as np

def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)

def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass
try:
    import ctypes

    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

def read_yaml_config(config_file):
    with open(config_file, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
            return None

def create_sphere_mask(shape, center, radius):
    Z, X, Y = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = (Y - center[2]) ** 2 + (X - center[1]) ** 2 + (Z - center[0]) ** 2
    mask = dist_from_center <= radius ** 2
    return th.tensor(mask, dtype=th.uint8)

def process_3d_mask_old(mask_3d, sigma=2, threshold=0.5):
    print("sum:", th.sum(mask_3d))
    if th.sum(mask_3d) < 20:
        sigma = 0.5
        threshold=0.01
    elif (th.sum(mask_3d) >= 20) and (th.sum(mask_3d) < 50):
        sigma = 0.5
        threshold=0.1
    elif (th.sum(mask_3d) >= 50) and (th.sum(mask_3d) < 100) :
        sigma = 0.5
        threshold=0.2
    elif (th.sum(mask_3d) >= 100) and (th.sum(mask_3d)<1000):
        sigma = 1 
        threshold=0.3
    elif (th.sum(mask_3d) >= 1000) and (th.sum(mask_3d)<10000):
        sigma = 1.5
    else :
        sigma = 2
    device = mask_3d.device
    mask_3d = mask_3d.cpu().numpy()
    filled_mask = binary_fill_holes(mask_3d)
    smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=sigma)
    smoothed_mask_binary = smoothed_mask > threshold
    labeled_mask, num_features = label(smoothed_mask_binary)
    region_sizes = np.bincount(labeled_mask.ravel())
    region_sizes[0] = 0  
    largest_region_label = region_sizes.argmax()
    largest_region_mask = (labeled_mask == largest_region_label)
    largest_region_mask = th.from_numpy(largest_region_mask).float().to(device)
    return largest_region_mask



def process_3d_mask(mask_3d, sigma=2, threshold=0.5):
    print("sum:", th.sum(mask_3d))
    total_sum = th.sum(mask_3d)
    if total_sum < 20:
        sigma = 0.5
        threshold = 0.01
    elif 20 <= total_sum < 50:
        sigma = 0.5
        threshold = 0.1
    elif 50 <= total_sum < 100:
        sigma = 0.5
        threshold = 0.2
    elif 100 <= total_sum < 1000:
        sigma = 1
        threshold = 0.3
    elif 1000 <= total_sum < 10000:
        sigma = 1.5
    else:
        sigma = 2
    device = mask_3d.device
    mask_3d = mask_3d.cpu().numpy()[0]  
    filled_mask = binary_fill_holes(mask_3d)
    smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=sigma)
    smoothed_mask_binary = smoothed_mask > threshold
    labeled_mask, num_features = label(smoothed_mask_binary)
    region_sizes = np.bincount(labeled_mask.ravel())
    region_sizes[0] = 0  
    largest_region_label = region_sizes.argmax()
    largest_region_mask = (labeled_mask == largest_region_label)
    interpolated_mask = zoom(largest_region_mask.astype(float), zoom=(1.5, 1.5, 1.5), order=1)
    interpolated_mask=np.expand_dims(interpolated_mask, axis=0)
    largest_region_mask = th.from_numpy(interpolated_mask > 0.5).float().to(device)
    return largest_region_mask

@hydra.main(config_path='confs', config_name='lidc_mask', version_base=None)
def main(conf: DictConfig):
    print("Start", conf['name'])
    device = dev(conf.get('device'))
    model = Unet3D(
        dim=conf.diffusion_img_size,
        dim_mults=conf.dim_mults,
        channels=conf.unet_num_channels,
        out_dim=conf.out_dim,  
        cond_dim=None,
    )
    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
        timesteps=conf.timesteps,
        loss_type=conf.loss_type,
    )
    diffusion.to(device)

    weights_dict = {}
    for k, v in (load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")["model"].items()):
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    diffusion.load_state_dict(weights_dict)

    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print("sampling...")
    dl = get_inference_dataloader(conf.dataset_root_dir, test_txt_path=conf.test_txt_path)  # batch_size默认写为1
    num_epoches = 1
    for _ in range(num_epoches):
        for batch in iter(dl):
            for k in batch.keys():
                if isinstance(batch[k], th.Tensor):
                    batch[k] = batch[k].to(device)
            model_kwargs = {}
            model_kwargs["gt"] = batch['GT']
            gt_name = batch['GT_name'][0]
            name_part, extension = gt_name.rsplit('.nii.gz', 1)
            gt_name = f"{name_part}.nii.gz"
            gt_keep_mask = batch.get('gt_keep_mask')
            if gt_keep_mask is not None:
                model_kwargs['gt_keep_mask'] = gt_keep_mask
            sample_fn = diffusion.p_sample_loop
            shape = (32, 64, 64)
            center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
            radius = 20
            small_sphere = create_sphere_mask(shape, center, radius)
            small_sphere = small_sphere * 2 - 1
            small_sphere = small_sphere.unsqueeze(0).unsqueeze(0).float()
            sphere = batch['sphere']
            mask_result = sample_fn(
                shape=(1, 1, 32, 64, 64), 
                sphere=sphere
            )
            mask_result = mask_result.squeeze(0).cpu()
            mask_result = (mask_result + 1) * 0.5
            mask_result = (mask_result > 0.5).float()
            mask_result = process_3d_mask(mask_result)
            restore_affine = batch['affine'].squeeze(0).cpu()
            name_part, extension = gt_name.rsplit('.nii.gz', 1)[0], '.nii.gz'
            main_name, vol_part = name_part.rsplit('_Vol_', 1)
            gen_mask_name = f"{main_name}_GenMask_{vol_part}{extension}"
            gen_mask = tio.LabelMap(tensor=mask_result, channels_last=False, affine=restore_affine)
            gen_mask.save(conf.gen_mask_path + gen_mask_name)

    print("sampling complete")


if __name__ == "__main__":
    main()