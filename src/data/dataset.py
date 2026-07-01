import os
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset as _hf_load_dataset

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .transforms import get_train_transforms, get_eval_transforms


class HFCIFAR100Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
        label = item["fine_label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class CIFAR100DataModule:
    def __init__(self, config, data_dir="./data"):
        self.config = config
        self.data_dir = data_dir
        self.batch_size = config.get("batch_size", 64)
        self.num_workers = 0
        self.pin_memory = config.get("pin_memory", True)
        self.train_split = config.get("train_split", 0.8)

        self.train_transforms = get_train_transforms()
        self.eval_transforms = get_eval_transforms()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._class_names = None

    def prepare_data(self):
        _hf_load_dataset("uoft-cs/cifar100", cache_dir=f"{self.data_dir}/hf_cache")

    def setup(self, stage=None):
        print("Loading CIFAR-100...", flush=True)
        ds = _hf_load_dataset("uoft-cs/cifar100", cache_dir=f"{self.data_dir}/hf_cache")
        print(f"Loaded: train={len(ds['train'])}, test={len(ds['test'])}", flush=True)

        self._class_names = ds["train"].features["fine_label"].names

        full_train = ds["train"]
        test_hf = ds["test"]

        total = len(full_train)
        train_size = int(self.train_split * total)
        val_size = total - train_size

        indices = list(range(total))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_subset = full_train.select(train_indices)
        val_subset = full_train.select(val_indices)

        self.train_dataset = HFCIFAR100Dataset(train_subset, transform=self.train_transforms)
        self.val_dataset = HFCIFAR100Dataset(val_subset, transform=self.eval_transforms)
        self.test_dataset = HFCIFAR100Dataset(test_hf, transform=self.eval_transforms)

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}", flush=True)

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self):
        return 100

    @property
    def class_names(self):
        return self._class_names
