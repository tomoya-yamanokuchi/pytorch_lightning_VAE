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


class Objective:
    def __init__(self, config):
        self.config = config


    def __call__(self, trial: optuna.trial.Trial) -> float:
        n_layers    = trial.suggest_int("n_layers", 1, 3)
        dropout     = trial.suggest_float("dropout", 0.2, 0.5)
        output_dims = [trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)]

        model = LightningNet(dropout, output_dims)
        datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

        trainer = pl.Trainer(
            logger=True,
            limit_val_batches=PERCENT_VALID_EXAMPLES,
            checkpoint_callback=False,
            max_epochs=EPOCHS,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        )
        hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        return trainer.callback_metrics["val_acc"].item()


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        study = optuna.create_study(direction="maximize")
        study.optimize(Objective(cfg), n_trials=100, timeout=600)

    get_config()