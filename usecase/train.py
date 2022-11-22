import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import hydra
from omegaconf import DictConfig, OmegaConf
from domain.train.Training import Training


# config_name = "config"
config_name = "config_dsvae"

@hydra.main(version_base=None, config_path="../conf", config_name=config_name)
def get_config(cfg: DictConfig) -> None:
    train = Training(cfg)
    train.run()

get_config()