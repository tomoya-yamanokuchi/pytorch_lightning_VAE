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
    def __init__(self, config):
        self.config = config


    def run(self, config_raytune):
        config = self.config
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
    from collections import defaultdict

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        train = Train(cfg)

        recursive_defaultdict = lambda: defaultdict(recursive_defaultdict)
        config = recursive_defaultdict()
        config["model"]["kld_weight"] = tune.choice([1e-2, 1e-3, 1e-4])
        config["model"]["conv_out_channels"] = tune.choice(
            [
                [32, 32, 32, 32],
                [32, 64, 64, 64],
                [32, 64, 128, 128],
                [32, 64, 128, 64]
            ]
        )

        for key, val in config.items():
            if isinstance(val, defaultdict):
                for key_nest, val_nest in val.items():
                    config[key][key_nest] = cfg[key][key_nest]
            else:
                config[key] = cfg[key]

        scheduler  = SchedulerFactory().create(**cfg.raytune.scheduler)
        # search_alg = SearchAlgorithmFactory().create(**cfg.raytune.search_algorithm, **cfg.raytune.common)
        # search_alg = SearchAlgorithmFactory().create(**cfg.raytune.search_algorithm)

        analysis = tune.run(train.run,
            config     = config,
            scheduler  = scheduler,
            # search_alg = search_alg,
            **cfg.raytune.general,
            **cfg.raytune.common,
        )
        print("Best hyperparameters found were: ", analysis.best_config)

    get_config()
