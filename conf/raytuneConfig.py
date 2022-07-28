from dataclasses import dataclass, field
from typing import Dict

@dataclass
class generalConfig:
    name               : str  = "tune_ABC"
    num_samples        : int  = 10
    fail_fast          : str  = "raise"
    resources_per_trial: Dict = field(default_factory=lambda: {'gpu': 1})

@dataclass
class commonConfig:
    metric : str = 'loss'
    mode   : str = 'min'

@dataclass
class schedulerConfig:
    name            : str = "async_hyperband"
    time_attr       : str = "training_iteration"
    max_t           : int = 1000
    grace_period    : int = 5
    reduction_factor: int = 2

@dataclass
class search_algorithmConfig:
    name          : str = "random"
    max_concurrent: int = 4


@dataclass # -----------------------------------------------------------
class raytuneConfig:
    general          : generalConfig          = generalConfig()
    common           : commonConfig           = commonConfig()
    scheduler        : schedulerConfig        = schedulerConfig()
    search_algorithm : search_algorithmConfig = search_algorithmConfig()


if __name__ == '__main__':
    import pprint
    from omegaconf import OmegaConf
    base_config = OmegaConf.structured(raytuneConfig)
    cli_config  = OmegaConf.from_cli()
    config      = OmegaConf.merge(base_config, cli_config)
    pprint.pprint(config)