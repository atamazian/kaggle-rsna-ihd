from typing import Union

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
import timm


class IHDModel(LightningModule):
    """Basic IHD model.
    >>> model = IHDModel("resnet18")
    """

    def __init__(
        self, model: Union[str, nn.Module], pretrained: bool = True, lr: float = 1e-4
    ):
        super().__init__()
        if isinstance(model, str):
            self.model = timm.create_model(
                model, pretrained=pretrained, num_classes=6, in_chans=1
            )
        else:
            self.model = model
        self.learn_rate = lr
        self.loss_fn = F.binary_cross_entropy_with_logits

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        self.log("valid_loss", loss, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )
        return [optimizer], [scheduler]
