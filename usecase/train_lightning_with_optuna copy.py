from distutils.command.config import config
from mimetypes import init
import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from model.ModelFactory import ModelFactory
from datamodule.DataModuleFactory import DataModuleFactory
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from omegaconf import OmegaConf
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import mlflow


import hydra
from omegaconf import DictConfig
@hydra.main(version_base=None, config_path="../conf", config_name="config_optuna")
def get_config(cfg: DictConfig) -> None:

    # n_layers    = trial.suggest_int("n_layers", 1, 3)
    # dropout     = trial.suggest_float("dropout", 0.2, 0.5)
    # output_dims = [trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)]

    # x: float = self.config.model.
    # y: float = self.config.y

    mlflow.pytorch.autolog()

    config      = cfg
    model       = ModelFactory().create(**config.model)
    tb_logger   = TensorBoardLogger(**config.logger)
    p           = pathlib.Path(tb_logger.log_dir)
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
            ),
            # PyTorchLightningPruningCallback(trial, monitor="val_loss")
        ],
        **config.trainer
    )

    # hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    # trainer.logger.log_hyperparams(hyperparameters)
    data = DataModuleFactory().create(**config.datamodule)
    with mlflow.start_run() as run:
        trainer.fit(model, datamodule=data)

    return trainer.callback_metrics["val_loss"].item()



get_config()

