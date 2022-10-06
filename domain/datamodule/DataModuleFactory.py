from .mnist.MNISTDataModule import MNISTDataModule
from .sprite.SpriteDataModule import SpriteDataModule


'''
データローダー直す
'''

class DataModuleFactory:
    def create(self, name: str, data_dir: str):
        name = name.lower()
        if   name == "mnist": return MNISTDataModule(data_dir)
        elif name == "sprite": return SpriteDataModule(data_dir)
        else: NotImplementedError()