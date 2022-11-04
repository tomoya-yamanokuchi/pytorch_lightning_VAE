import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import numpy as np
import torchinfo
import torch
from torchvision import utils
from torch import Tensor
from torch import optim
from typing import List, Any
import pytorch_lightning as pl
from .DisentangledSequentialVariationalAutoencoder import DisentangledSequentialVariationalAutoencoder
from .. import visualization

import cv2
from custom.utility.image_converter import torch2numpy

class LitDisentangledSequentialVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                kld_weight,
                **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kld_weight = kld_weight
        self.model = DisentangledSequentialVariationalAutoencoder(**kwargs)
        # self.summary = torchinfo.summary(self.model, input_size=(131, 8, 3, 64, 64))


    def forward(self, input, **kwargs) -> Any:
        return self.model.forward(input)


    def decode(self, z, f):
        '''
        input:
            - z: shape = []
            - f: shape = []
        '''
        # import ipdb; ipdb.set_trace()

        num_batch, step, _ = z.shape
        # z         = z.view(num_batch, step, -1)
        # import ipdb; ipdb.set_trace()
        # f         = f.view(num_batch, step, -1)
        x_recon   = self.model.frame_decoder(torch.cat((z, f.unsqueeze(1).expand(num_batch, step, -1)), dim=2))
        return x_recon


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def training_step(self, batch, batch_idx):
        # print("batch_idx: ", batch_idx)
        index, img_batch = batch
        results_dict     = self.model.forward(img_batch)
        loss             = self.model.loss_function(
            **results_dict,
            x         = img_batch,
            M_N       = self.kld_weight,
            batch_idx = batch_idx
        )
        # self.save_progress(img_batch, results_dict)
        self.log("val_loss", loss["loss"])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)
        return loss['loss']


    def validation_step(self, batch, batch_idx):
        index, img_batch = batch  # shape = [num_batch, step, channel, w, h], Eg.) [128, 8, 3, 64, 64])
        results_dict     = self.model.forward(img_batch)
        loss             = self.model.loss_function(
            **results_dict,
            x         = img_batch,
            M_N       = self.kld_weight,
            batch_idx = batch_idx
        )
        self.log("val_loss", loss["loss"])
        if batch_idx == 0:
            self.save_progress(img_batch, results_dict)


    def save_progress(self, img_batch, results_dict: dict):
        if pathlib.Path(self.logger.log_dir).exists():
            p = pathlib.Path(self.logger.log_dir + "/reconstruction"); p.mkdir(parents=True, exist_ok=True)
            num_batch, step, channel, width, height = img_batch.shape

            # cv2.imshow("img", torch2numpy(img_batch[0, 0]) / 255. )
            # cv2.imshow("img", torch2numpy(results_dict["x_recon"][0, 0]) / 255. )
            # cv2.waitKey(100)
            # import ipdb; ipdb.set_trace()

            save_sequence = 8  # np.minimum(10, mod)
            images        = []
            permute       = [2, 1, 0] # BGR --> RGB for accurate save using PIL
            for n in range(save_sequence):
                images.append(utils.make_grid(results_dict["x_recon"][n], nrow=step))
                images.append(utils.make_grid(              img_batch[n], nrow=step))

            # save input and reconstructed images
            utils.save_image(
                tensor = torch.cat(images, dim=1),
                fp     = os.path.join(str(p), 'reconstruction_epoch' + str(self.current_epoch)) + '.png',
            )
