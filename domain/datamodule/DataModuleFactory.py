from .mnist.MNISTDataModule import MNISTDataModule


class DataModuleFactory:
    def create(self, name: str, data_dir: str):
        name = name.lower()
        if   name == "mnist": return MNISTDataModule(data_dir)
        elif name == "sprite": return MNISTDataModule(data_dir)
        else: NotImplementedError()