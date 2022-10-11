import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
from custom.utility import image_converter
import os
from torchvision import utils

import cv2
import numpy as np
cv2.namedWindow('img', cv2.WINDOW_NORMAL)


# config
reload_dir         = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_98"
config             = OmegaConf.load(reload_dir + "/config.yaml")
config.reload.path = reload_dir + "/checkpoints/last.ckpt"

# model loading
device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lit_model_class = ModelFactory().create(config.model.name)
# lit_model       = lit_model_class(**config.model)

# print(lit_model.model.state_dict()["frame_decoder.deconv_fc.0.model.1.weight"])
# import ipdb; ipdb.set_trace()

lit_model = lit_model_class.load_from_checkpoint(config.reload.path)
lit_model.freeze()
model     = lit_model.eval().cuda(device)

print(lit_model.model.state_dict()["frame_decoder.deconv_fc.0.model.1.weight"])

import ipdb; ipdb.set_trace()

# data loading
datamodule = DataModuleFactory().create(**config.datamodule)
datamodule.setup(stage="fit")
data       = datamodule.val_dataloader()


for batch in data:
    batch = batch.to(device)
    import ipdb; ipdb.set_trace()
    # batch = batch[:2]
    return_dict = model(batch)
    # x_recon = model.decode(return_dict["z_sample"], return_dict["f_sample"])
    x_recon = return_dict["x_recon"]


    # import ipdb; ipdb.set_trace()

    save_sequence = 10
    step          = 8
    images        = []
    for n in range(save_sequence):
        images.append(utils.make_grid(return_dict["x_recon"][n], nrow=step))
        images.append(utils.make_grid(                batch[n], nrow=step))

    # 入力画像と再構成画像を並べて保存
    utils.save_image(
        tensor = torch.cat(images, dim=1),
        fp     = "/home/tomoya-y/workspace/pytorch_lightning_VAE/reconstruction_epoch_test.png",
    )

    # import ipdb; ipdb.set_trace()

    # num_batch, step, channel, width, height =  x_recon.shape
    # for n in range(num_batch):
    #     for t in range(step):
    #         x   = image_converter.torch2numpy(x_recon[n,t])
    #         img = image_converter.torch2numpy(images[n,t])

    #         # import ipdb; ipdb.set_trace()
    #         cv2.imshow('img', np.concatenate((x, img), axis=1))
    #         cv2.waitKey(10)