import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from dataclasses import dataclass, field
from .experimentConfig import experimentConfig
from .checkpointConfig import checkpointConfig
from .datamoduleConfig import datamoduleConfig
from .modelConfig import modelConfig
from .raytuneConfig import raytuneConfig


@dataclass
class Config:
    experiment : experimentConfig = experimentConfig()
    checkpoint : checkpointConfig = checkpointConfig()
    datamodule : datamoduleConfig = datamoduleConfig()
    model      : modelConfig      = modelConfig()
    raytune    : raytuneConfig    = raytuneConfig()


if __name__ == '__main__':
    import pprint
    from omegaconf import OmegaConf
    base_config = OmegaConf.structured(Config)
    cli_config  = OmegaConf.from_cli()
    config      = OmegaConf.merge(base_config, cli_config)
    pprint.pprint(config)