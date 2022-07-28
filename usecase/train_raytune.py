import os
import hydra
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from domain.raytune.SchedulerFactory import SchedulerFactory
from domain.raytune.SearchAlgorithmFactory import SearchAlgorithmFactory
from Training import Training


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def get_config(cfg: DictConfig) -> None:
    recursive_defaultdict = lambda: defaultdict(recursive_defaultdict)
    config                = recursive_defaultdict()
    config["model"]["kld_weight"] = tune.choice([1e-2, 1e-3, 1e-4])
    config["model"]["conv_out_channels"] = tune.choice(
        [
            [32, 32, 32, 32],
            [32, 64, 64, 64],
            [32, 64, 128, 128],
            [32, 64, 128, 64]
        ]
    )

    train = Training(cfg)

    scheduler  = SchedulerFactory().create(**cfg.raytune.scheduler)
    search_alg = SearchAlgorithmFactory().create(**cfg.raytune.search_algorithm, **cfg.raytune.common)
    analysis   = tune.run(train.run,
        config     = config,
        scheduler  = scheduler,
        search_alg = search_alg,
        **cfg.raytune.general,
        **cfg.raytune.common,
    )
    print("Best hyperparameters found were: ", analysis.best_config)

get_config()