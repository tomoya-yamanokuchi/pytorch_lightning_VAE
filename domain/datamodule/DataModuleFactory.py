from .mnist.MNISTDataModule import MNISTDataModule
from .sprite.SpriteDataModule import SpriteDataModule
from .robel_claw_valve.ActionNormalizedValveDataModule import ActionNormalizedValveDataModule


'''
データローダー直す
'''

class DataModuleFactory:
    def create(self, name: str, data_dir: str):
        name = name.lower()
        if   name == "mnist"            : return MNISTDataModule(data_dir)
        elif name == "sprite"           : return SpriteDataModule(data_dir)
        elif name == "action_norm_valve": return ActionNormalizedValveDataModule(data_dir)
        else: NotImplementedError()