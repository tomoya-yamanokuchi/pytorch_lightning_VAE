import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory


# config_name = "config"
config_name = "config_dsvae"



@hydra.main(version_base=None, config_path="../conf", config_name=config_name)
def get_config(cfg: DictConfig) -> None:
    config = cfg
    config.reload.path = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_75/checkpoints/epoch=0.ckpt"


    model = ModelFactory().create(**config.model)
    model.load_from_checkpoint(config.reload.path)
    model.freeze()


    datamodule  = DataModuleFactory().create(**config.datamodule)

    trainer     = Trainer()

    import ipdb; ipdb.set_trace()
    # test_sample = next(iter(datamodule.predict_dataloader()))

    return_dict = trainer.predict(model, datamodule)




    import ipdb; ipdb.set_trace()


get_config()