import logging
import multiprocessing as mproc
import os
from math import ceil

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from monai import transforms as T

TRAIN_TRANSFORM = T.Compose(
    [
        T.AddChannel(),
        T.CenterSpatialCrop((200, 200)),
        T.RandFlip(prob=0.5, spatial_axis=0),
        T.ScaleIntensity(),
        T.EnsureType(),
    ]
)

VALID_TRANSFORM = T.Compose(
    [
        T.AddChannel(),
        T.CenterSpatialCrop((200, 200)),
        T.ScaleIntensity(),
        T.EnsureType(),
    ]
)


class IHDDataset(Dataset):
    def __init__(
        self,
        path_csv: str,
        path_img_dir: str,
        transforms=None,
        mode: str = "train",
        split: float = 0.8,
    ):
        self.path_img_dir = path_img_dir
        self.transforms = transforms
        self.mode = mode

        self.data = pd.read_csv(path_csv)
        self.data["image_id"] = self.data["ID"].apply(
            lambda x: "_".join(x.split("_")[:-1]) + ".png"
        )
        self.data["type"] = self.data["ID"].apply(lambda x: x.split("_")[2])
        self.data = (
            self.data[["Label", "image_id", "type"]]
            .drop_duplicates()
            .pivot(index="image_id", columns="type", values="Label")
            .reset_index()
        )
        label_cols = [
            "epidural",
            "intraparenchymal",
            "intraventricular",
            "subarachnoid",
            "subdural",
            "any",
        ]

        # shuffle data
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # split dataset
        assert 0.0 <= split <= 1.0
        frac = int(ceil(split * len(self.data)))
        self.data = self.data[:frac] if mode == "train" else self.data[frac:]
        self.img_names = list(self.data["image_id"])
        self.labels = list(self.data[label_cols].values)

    def __getitem__(self, idx: int) -> tuple:
        img_path = os.path.join(self.path_img_dir, self.img_names[idx])
        assert os.path.isfile(img_path)
        label = self.labels[idx]
        img = T.LoadImage(image_only=True)(img_path)

        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self) -> int:
        return len(self.data)


class IHDDataModule(LightningDataModule):
    def __init__(
        self,
        path_csv: str,
        path_img_dir: str,
        train_transform=TRAIN_TRANSFORM,
        valid_transform=VALID_TRANSFORM,
        batch_size: int = 128,
        split: float = 0.8,
    ):
        super().__init__()
        self.path_csv = path_csv
        self.path_img_dir = path_img_dir
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.batch_size = batch_size
        self.split = split

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = IHDDataset(
            self.path_csv,
            self.path_img_dir,
            split=self.split,
            mode="train",
            transforms=self.train_transform,
        )
        logging.info(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = IHDDataset(
            self.path_csv,
            self.path_img_dir,
            split=self.split,
            mode="valid",
            transforms=self.valid_transform,
        )
        logging.info(f"validation dataset: {len(self.valid_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=mproc.cpu_count(),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=mproc.cpu_count(),
            shuffle=False,
        )

    def test_dataloader(self):
        pass
