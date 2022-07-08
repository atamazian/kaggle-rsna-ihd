import os
import numpy
from kaggle_rsna_ihd.datasets import IHDDataModule, IHDDataset
from demo import DEMO_DIR

print(DEMO_DIR)

def test_dataset(path_data=DEMO_DIR):
    dataset = IHDDataset(
        path_csv=os.path.join(path_data, "train.csv"),
        path_img_dir=os.path.join(path_data, "train_images"),
    )
    img, lb = dataset[0]
    assert isinstance(img, numpy.ndarray)


def test_datamodule(path_data=DEMO_DIR):
    dm = IHDDataModule(
        path_csv=os.path.join(path_data, "train.csv"),
        path_img_dir=os.path.join(path_data, "train_images"),
    )
    dm.setup()

    for imgs, lbs in dm.train_dataloader():
        assert len(imgs)
        assert len(lbs)
        break