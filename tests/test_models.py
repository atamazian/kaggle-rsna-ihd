import os
from pytorch_lightning import Trainer
from kaggle_rsna_ihd.datasets import IHDDataModule
from kaggle_rsna_ihd.models import IHDModel
from demo import DEMO_DIR 

def test_model(tmpdir, path_data=DEMO_DIR):
    dm = IHDDataModule(
        path_csv=os.path.join(path_data, "train.csv"),
        path_img_dir=os.path.join(path_data, "train_images"),
        batch_size=1,
        split=0.6,
    )
    model = IHDModel(model="resnet18")

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        gpus=0,
    )
    dm.setup()
    trainer.fit(model, datamodule=dm)