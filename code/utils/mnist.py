"""MNIST loading and train/validation/test data splits."""

from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def _take(dataset, count):
    if count is None:
        return dataset
    if count < 1:
        raise ValueError("Sample counts must be positive or null.")
    return Subset(dataset, range(min(count, len(dataset))))


class MNISTDataModule(pl.LightningDataModule):
    """MNIST with a fixed train/validation split and optional sample limits."""

    def __init__(
        self,
        data_dir="data/mnist",
        batch_size=64,
        train_samples=None,
        val_samples=None,
        test_samples=None,
        image_size=32,
        split_seed=42,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = Path(to_absolute_path(data_dir))
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.split_seed = split_seed
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            full_train = MNIST(self.data_dir, train=True, transform=self.transform)
            train_split, val_split = random_split(
                full_train,
                [55_000, 5_000],
                generator=torch.Generator().manual_seed(self.split_seed),
            )
            self.train_ds = _take(train_split, self.train_samples)
            self.val_ds = _take(val_split, self.val_samples)

        if stage in (None, "test", "predict"):
            full_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_ds = _take(full_test, self.test_samples)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
