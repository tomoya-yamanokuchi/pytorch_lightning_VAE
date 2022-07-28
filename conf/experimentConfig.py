from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class loggerConfig:
  save_dir: str = "logs/"
  name    : str = "VAE"

@dataclass
class trainerConfig:
  gpus                : List[int] = field(default_factory=lambda: [0])
  limit_train_batches : int       = 100
  max_epochs          : int       = 30

@dataclass
class experimentConfig:
    manual_seed : int           = 1265
    trainer     : trainerConfig = trainerConfig()
    logger      : loggerConfig  = loggerConfig()


if __name__ == '__main__':
    import pprint
    from omegaconf import OmegaConf
    base_config = OmegaConf.structured(experimentConfig)
    cli_config  = OmegaConf.from_cli()
    config      = OmegaConf.merge(base_config, cli_config)
    pprint.pprint(config)