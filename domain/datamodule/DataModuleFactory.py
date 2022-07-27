
from .MNISTDataModule import MNISTDataModule


class DataModuleFactory:
    def create(self, name: str, data_dir: str):
        name = name.lower()
        if name == "mnist": return MNISTDataModule(data_dir)