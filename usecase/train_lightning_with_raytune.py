import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from omegaconf import OmegaConf
from ray.tune.integration.pytorch_lightning import TuneReportCallback


class Train:
    def __init__(self, config):
        self.config = config

    def run(self, config_raytune):

        config = self.config


        config.trainer.max_epochs = config_raytune["max_epochs"]


        seed = seed_everything(config.experiment.manual_seed, True)
        print("seed: ", seed)

        model = ModelFactory().create(**config.model)

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=config.logger.save_dir)
        p = pathlib.Path(tb_logger.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, tb_logger.log_dir + "/config.yaml")

        metrics = {"loss": "loss"}

        trainer = pl.Trainer(
            logger    = tb_logger,
            callbacks = [
                LearningRateMonitor(),
                ModelCheckpoint(
                    dirpath  = os.path.join(tb_logger.log_dir , "checkpoints"),
                    filename = '{epoch}',
                    **config.checkpoint
                ),
                TuneReportCallback(metrics, on="validation_end"),
            ],
            **config.trainer
        )

        data = DataModuleFactory().create(**config.datamodule)
        trainer.fit(model=model, datamodule=data)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    from ray import tune

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        train = Train(cfg)
        # train.run(cfg)

        config = {
            "max_epochs": tune.choice([1, 2, 3]),
        }

        analysis = tune.run(train.run,
            metric      = "loss",
            mode        = "min",
            config      = config,
            num_samples = 10,
            name        = "tune_mnist",
            fail_fast   = "raise",
            resources_per_trial = {'cpu':8, 'gpu':1},
        )
        print("Best hyperparameters found were: ", analysis.best_config)

    get_config()
