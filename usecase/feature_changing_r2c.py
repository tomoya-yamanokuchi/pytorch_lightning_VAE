import sys; import pathlib
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.test.TestModel import TestModel
from custom.utility.image_converter import torch2numpy
import os
from torchvision import utils

import cv2
import numpy as np

import cv2
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

test = TestModel(
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_205",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_202",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/R2C-DSVAE/version_28",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/R2C-DSVAE/version_32",
    config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/R2C-DSVAE/version_36",
    # checkpoints = "epoch=199.ckpt"
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()


iter_dataloader = iter(dataloader)
index, img_batch_can, img_batch_ran = next(iter_dataloader)
assert index[0] == 0

# ----------------------------
test_index = 17
# ----------------------------
img_seq         = img_batch_can[test_index].unsqueeze(dim=0).to(device)
return_dict_seq = model(img_seq)
x_recon         = model.decode(return_dict_seq["z_mean"], return_dict_seq["f_mean"])

# import ipdb; ipdb.set_trace()
step          = 25 # valve
num_slice     = 1
# step = 0

z = return_dict_seq["z_mean"]
f = return_dict_seq["f_mean"]

for m in range(1000):
    # for t in range(step):
    x_recon = model.decode(z, f)
    img     = utils.make_grid(x_recon[0][0], nrow=1, normalize=True)
    img     = torch2numpy(img)
    img     = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)
    cv2.waitKey(50)
    # import ipdb; ipdb.set_trace()
    # z += torch.randn_like(z) * 0.1
    f += torch.randn_like(f) * 0.1


