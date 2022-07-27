from distutils.command.config import config
from mimetypes import init
import os
import sys; import pathlib

from cv2 import DFT_INVERSE; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
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


class Objective:
    def __init__(self, config):
        self.config = config

    def __call__(self, trial: optuna.trial.Trial) -> float:
        # n_layers    = trial.suggest_int("n_layers", 1, 3)
        # dropout     = trial.suggest_float("dropout", 0.2, 0.5)
        # output_dims = [trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)]
        kld_weight = trial.suggest_categorical("kld_weight", [0.01, 0.001, 0.0001])
        self.config.model.kld_weight = kld_weight

        # mlflow.pytorch.autolog()
        # for key, val in self.config.model.items():
        #     mlflow.log_param(key, val)

        config      = self.config
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
                PyTorchLightningPruningCallback(trial, monitor="val_loss")
            ],
            **config.trainer
        )

        # hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
        # trainer.logger.log_hyperparams(hyperparameters)
        data = DataModuleFactory().create(**config.datamodule)

        with mlflow.start_run() as run:
            trainer.fit(model, datamodule=data)

        return trainer.callback_metrics["val_loss"].item()


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(Objective(cfg), n_trials=5, timeout=600)

        print('best_params', study.best_params)
        print('-1 x best_value', -study.best_value)

        print('\n --- sorted --- \n')
        sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
        for i, k in sorted_best_params:
            print(i + ' : ' + str(k))

    get_config()