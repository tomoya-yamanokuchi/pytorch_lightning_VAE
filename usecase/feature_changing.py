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
    config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_232",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_235",
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()


iter_dataloader = iter(dataloader)
index, img_batch = next(iter_dataloader)
assert index[0] == 0

# ----------------------------
test_index = 29  # 黒髪左歩き
# test_index = 16 # 緑髪右歩き
# ----------------------------
img_seq         = img_batch[test_index].unsqueeze(dim=0).to(device)
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


    '''
    dynamic info
    '''
    z[:, :, 6] += torch.randn_like(z[:, :, 6])
    z[:, :, 7] += torch.randn_like(z[:, :, 7])
    z[:, :, 12] += torch.randn_like(z[:, :, 12])

    '''
    color info
    '''
    # z[:, :, 0:6]  += torch.randn_like(z[:, :, 0:6])
    # z[:, :, 8:12] += torch.randn_like(z[:, :, 8:12])
    # z[:, :, 13:]  += torch.randn_like(z[:, :, 13:])


    # f += torch.randn_like(f)
    # f[:, 10:30] += torch.randn_like(f[:, 10:30])
    # f[:, :30] += torch.randn_like(f[:, :30])
    # f[:, 31:43] += torch.randn_like(f[:, 31:43])
    # f[:, 49:] += torch.randn_like(f[:, 49:])