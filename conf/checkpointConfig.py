from dataclasses import dataclass, field
from typing import Dict

@dataclass
class checkpointConfig:
  every_n_epochs          : int  = 1
  monitor                 : str  = "Reconstruction_Loss"
  save_last               : bool = False
  save_on_train_epoch_end : bool = True


if __name__ == '__main__':
    import pprint
    from omegaconf import OmegaConf
    base_config = OmegaConf.structured(checkpointConfig)
    cli_config  = OmegaConf.from_cli()
    config      = OmegaConf.merge(base_config, cli_config)
    pprint.pprint(config)