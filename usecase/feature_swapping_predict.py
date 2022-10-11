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
import os
from torchvision import utils

import cv2
import numpy as np



test = TestModel(
    config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_91",
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()


for batch in dataloader:
    import ipdb; ipdb.set_trace()
    batch       = batch.to(device)
    return_dict = model(batch)
    x_recon     = model.decode(return_dict["z_sample"], return_dict["f_sample"])
    # x_recon     = return_dict["x_recon"]

    save_sequence = 10
    step          = 8
    images        = []
    for n in range(save_sequence):
        images.append(utils.make_grid(return_dict["x_recon"][n], nrow=step))
        images.append(utils.make_grid(                 batch[n], nrow=step))

    # 入力画像と再構成画像を並べて保存
    utils.save_image(
        tensor = torch.cat(images, dim=1),
        fp     = "/home/tomoya-y/workspace/pytorch_lightning_VAE/reconstruction_epoch_test.png",
    )