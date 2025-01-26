from torch.utils.data import DataLoader
from dataset import LIDCMASKDataset, LIDCMASKInDataset

def get_inference_dataloader(dataset_root_dir, test_txt_path, batch_size=1, drop_last=False):
    train_dataset = LIDCMASKInDataset(root_dir=dataset_root_dir, test_txt_path=test_txt_path)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last)
    return loader


def get_train_dataset(cfg):
    train_dataset = LIDCMASKDataset(root_dir=cfg.dataset.root_dir, text_txt_path=cfg.dataset.test_txt_dir)
    sampler = None
    return train_dataset, sampler



