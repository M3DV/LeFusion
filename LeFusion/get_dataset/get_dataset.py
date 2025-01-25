from torch.utils.data import DataLoader
from dataset import LIDCDataset, LIDCInDataset
from dataset import EMIDECDataset, EMIDECInDataset


def get_inference_dataloader(dataset_root_dir, test_txt_dir,batch_size=1, drop_last=False, data_type=''):
    if data_type == 'lidc':
        train_dataset = LIDCInDataset(root_dir=dataset_root_dir, test_txt_dir=test_txt_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'emidec':
        train_dataset = EMIDECInDataset(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    return loader

def get_train_dataset(cfg):
    if cfg.dataset.data_type == 'lidc':
        train_dataset = LIDCDataset(root_dir=cfg.dataset.root_dir, test_txt_dir=cfg.dataset.test_txt_dir)
        sampler = None
    elif cfg.dataset.data_type == 'emidec':
        train_dataset = EMIDECDataset(root_dir=cfg.dataset.root_dir)
        sampler = None
    return train_dataset, sampler


