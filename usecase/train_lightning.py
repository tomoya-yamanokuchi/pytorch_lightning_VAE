import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from model.ModelFactory import ModelFactory
from datamodule.DataModuleFactory import DataModuleFactory
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from omegaconf import OmegaConf


class Train:
    def run(self, config):
        seed = seed_everything(config.experiment.manual_seed, True)
        print("seed: ", seed)

        model     = ModelFactory().create(**config.model)
        tb_logger = TensorBoardLogger(**config.logger)
        p         = pathlib.Path(tb_logger.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, tb_logger.log_dir + "/config.yaml")

        trainer = pl.Trainer(
            logger    = tb_logger,
            callbacks = [
                LearningRateMonitor(),
                ModelCheckpoint(
                    dirpath  = os.path.join(tb_logger.log_dir , "checkpoints"),
                    filename = '{epoch}',
                    **config.checkpoint
                )
            ],
            **config.trainer
        )

        data = DataModuleFactory().create(**config.datamodule)
        trainer.fit(model=model, datamodule=data)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        train = Train()
        train.run(cfg)

    get_config()