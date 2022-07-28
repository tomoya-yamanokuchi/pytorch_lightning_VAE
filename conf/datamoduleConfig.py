from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class datamoduleConfig:
    name       : str = "mnist"
    data_dir   : str = "./data/"
    # batch_size : 256
    # shuffle    : False
    # num_workers: 0
    # pin_memory : False


if __name__ == '__main__':
    import pprint
    from omegaconf import OmegaConf
    base_config = OmegaConf.structured(datamoduleConfig)
    cli_config  = OmegaConf.from_cli()
    config      = OmegaConf.merge(base_config, cli_config)
    pprint.pprint(config)