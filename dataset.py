"""
Dataset utilities for the Flow Matching assignment
Students can use these utilities to load and preprocess the Simpsons dataset.
"""

import os
import time

# Set the kagglehub cache directory to ./data
os.environ["KAGGLEHUB_CACHE"] = "./data"

from itertools import chain
import random
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.utils import tensor_to_pil_image

# Note: kagglehub is required for downloading the dataset
# Install with: pip install kagglehub
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    raise ImportError("kagglehub not available. Dataset download will not work. Run `pip install kagglehub`.")


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

class SimpsonsDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        with open(os.path.join("data", f"{split}_split.txt"), "r") as f:
            self.image_paths = [os.path.join(root, f.strip()+".png") for f in f.readlines()]

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.image_paths)

class SimpsonsDataModule(object):
    def __init__(self, root="./data/datasets/kostastokis/simpsons-faces/versions/1/cropped", batch_size=32, num_workers=4, transform=None):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.image_resolution = 64

        if not os.path.exists(self.root):
            self._download_dataset(dir_path=self.root)
        self._set_dataset()

    def _set_dataset(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_resolution, self.image_resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        self.train_ds = SimpsonsDataset(self.root, "train", self.transform)
        self.val_ds = SimpsonsDataset(self.root, "val", self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)

    def _download_dataset(self, dir_path):
        os.environ["KAGGLEHUB_CACHE"] = "./data"
        try:
            import torch_xla.core.xla_model as xm
            is_xla = True
        except Exception:
            xm = None
            is_xla = False
        if not os.path.exists(dir_path):
            if is_xla:
                master = xm.is_master_ordinal()
                if master:
                    print("Downloading dataset...")
                    try:
                        kagglehub.dataset_download("kostastokis/simpsons-faces")
                    except FileNotFoundError:
                        time.sleep(1)
                        kagglehub.dataset_download("kostastokis/simpsons-faces")
                try:
                    xm.rendezvous("dataset_ready")
                except Exception:
                    pass
                wait_until = time.time() + 300
                while not os.path.exists(dir_path) and time.time() < wait_until:
                    time.sleep(1)
            else:
                print("Downloading dataset...")
                kagglehub.dataset_download("kostastokis/simpsons-faces")
        else:
            print("Dataset already downloaded")

if __name__ == "__main__":
    data_module = SimpsonsDataModule()
    print(f"# training images: {len(data_module.train_ds)}")
    print(f"# validation images: {len(data_module.val_ds)}")