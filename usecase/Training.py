import os
import copy
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from omegaconf import OmegaConf
from collections import defaultdict


class Training:
    def __init__(self, config, additionl_callbacks=[]):
        self.config              = config
        self.additionl_callbacks = additionl_callbacks


    def _override_config(self, config_raytune):
        config = copy.deepcopy(self.config)
        for key, val in config_raytune.items():
            if isinstance(val, defaultdict):
                for key_nest, val_nest in val.items():
                    config[key][key_nest] = config_raytune[key][key_nest]
            else:
                config[key] = config_raytune[key]
        return config


    def run(self, config_raytune=None):
        if   config_raytune is None: config = self.config
        else:                        config = self._override_config(config_raytune)
        self._run(config)


    def _run(self, config):
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
            ] + self.additionl_callbacks,
            **config.trainer
        )

        data = DataModuleFactory().create(**config.datamodule)
        trainer.fit(model=model, datamodule=data)