import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from .Sprite import Sprite


class SpriteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        # download
        print("no implementation for download data")

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = Sprite(self.data_dir, train=True, transform=self.transform)
            import ipdb; ipdb.set_trace()
            self.train, self.val = random_split(full, [9000, 1000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = Sprite(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.predict = Sprite(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=5000)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=128)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=128)

    @property
    def transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize(size=(64)),
                # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )


if __name__ == '__main__':
    mnist = SpriteDataModule("./")
