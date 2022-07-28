import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from domain.raytune.SchedulerFactory import SchedulerFactory
from domain.raytune.SearchAlgorithmFactory import SearchAlgorithmFactory

class Train:
    # def __init__(self, config):
    #     self.config = config

    def run(self, config):
        # config = self.config
        # config.trainer.max_epochs = config_raytune["max_epochs"]
        # config.model.kld_weight        = config_raytune["kld_weight"]
        # config.model.conv_out_channels = config_raytune["conv_out_channels"]
        # config.model.in_channels       = config_raytune["in_channels"]

        seed = seed_everything(config.experiment.manual_seed, True)
        print("seed: ", seed)

        model = ModelFactory().create(**config.model)

        tb_logger = TensorBoardLogger(**config.logger)
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
                TuneReportCallback(metrics, on="validation_end")
            ],
            **config.trainer
        )

        data = DataModuleFactory().create(**config.datamodule)
        trainer.fit(model=model, datamodule=data)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from ray import tune
    from conf.Config import Config

    base_config = OmegaConf.structured(Config)
    cli_config  = OmegaConf.from_cli()
    config      = OmegaConf.merge(base_config, cli_config)

    scheduler  = SchedulerFactory().create(**config.raytune.scheduler)
    # search_alg = SearchAlgorithmFactory().create(**cfg.raytune.search_algorithm, **cfg.raytune.common)
    # search_alg = SearchAlgorithmFactory().create(**cfg.raytune.search_algorithm)

    train = Train()
    analysis = tune.run(train.run,
        config     = config,
        scheduler  = scheduler,
        # search_alg = search_alg,
        **config.raytune.general,
        **config.raytune.common,
    )
    print("Best hyperparameters found were: ", analysis.best_config)