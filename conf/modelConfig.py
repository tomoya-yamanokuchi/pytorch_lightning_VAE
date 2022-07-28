from dataclasses import dataclass, field
from typing import Dict, List, Any
from ray import tune

# @dataclass
# class modelConfig:
#     name             : str       = "vae"
#     in_channels      : int       = 3
#     conv_out_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
#     latent_dim       : int       = 2
#     kld_weight       : float     = 0.0001


@dataclass
class modelConfig:
    name             : str       = "vae"
    in_channels      : int       = 3
    conv_out_channels       = tune.choice(
        [
            [32, 32, 32, 32],
            [32, 64, 64, 64],
            [32, 64, 128, 128],
            [32, 64, 128, 64]
        ]
    )
    latent_dim       : int       = 2
    kld_weight = tune.choice([1e-2, 1e-3, 1e-4])


if __name__ == '__main__':
    import pprint
    from omegaconf import OmegaConf
    base_config = OmegaConf.structured(modelConfig)
    cli_config  = OmegaConf.from_cli()
    config      = OmegaConf.merge(base_config, cli_config)
    pprint.pprint(config)