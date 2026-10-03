import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models import resnet18


class LitResNet18(pl.LightningModule):
    """Lightning ResNet-18 model for MNIST image classification."""

    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.network = resnet18(weights=None, num_classes=num_classes)

    def forward(self, images):
        return self.network(images)

    def _shared_step(self, batch, stage):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        if stage == "train":
            # Live progress is local; epoch metrics are synchronized across ranks.
            self.log(
                "train_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False,
                logger=False,
            )
            self.log(
                "train_acc_step", accuracy, prog_bar=True, on_step=True, on_epoch=False,
                logger=False,
            )
        self.log(
            f"{stage}_loss", loss, prog_bar=stage != "train", on_step=False,
            on_epoch=True, sync_dist=True,
        )
        self.log(
            f"{stage}_acc", accuracy, prog_bar=stage != "train", on_step=False,
            on_epoch=True, sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
