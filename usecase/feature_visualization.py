import copy
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.test.TestModel import TestModel
from custom.utility import image_converter
import numpy as np

from domain.visualize.vector_heatmap import VectorHeatmap


test = TestModel(
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_205",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_202",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_235",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_232",
    config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_306",
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()


vectorHeatmap = VectorHeatmap()
for index, img, img_aug_context, img_aug_dynamics in dataloader:
    z = []
    for test_index in range(len(img)):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img)))

        img_seq     = img[test_index].unsqueeze(dim=0).to(device)
        return_dict = model(img_seq)
        # _f          = return_dict["f_mean"].to("cpu").numpy()
        _z          = return_dict["z_mean"].to("cpu").numpy()
        _, step, dim_z = _z.shape
        # print(np.linalg.norm(_z))
        z.append(copy.deepcopy(_z.reshape(step, dim_z).transpose()))

        # import ipdb; ipdb.set_trace()
        # vector_heatmap(np.concatenate(z, axis=0))
        vectorHeatmap.pause_show(_z.reshape(step, dim_z).transpose(), interval=0.05)


