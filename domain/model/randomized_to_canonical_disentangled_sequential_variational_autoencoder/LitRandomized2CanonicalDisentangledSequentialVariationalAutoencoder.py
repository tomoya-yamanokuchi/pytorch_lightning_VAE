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
from .Randomized2CanonicalDisentangledSequentialVariationalAutoencoder import Randomized2CanonicalDisentangledSequentialVariationalAutoencoder
from .. import visualization

import cv2
from custom.utility.image_converter import torch2numpy


class LitRandomized2CanonicalDisentangledSequentialVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                loss_weight,
                **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.loss_weight = loss_weight
        self.model       = Randomized2CanonicalDisentangledSequentialVariationalAutoencoder(**kwargs)
        self.summary = torchinfo.summary(self.model, input_size=(131, 8, 3, 64, 64))
        # import ipdb; ipdb.set_trace()


    def forward(self, input, **kwargs) -> Any:
        return self.model.forward(input)


    def decode(self, z, f):
        num_batch, step, _   = z.shape
        a_mean_decoded, _, _ = self.model.latent_frame_decoder(torch.cat((z, f.unsqueeze(1).expand(num_batch, step, -1)), dim=2))
        x_recon              = self.model.frame_decoder(a_mean_decoded)
        # import ipdb; ipdb.set_trace()
        return x_recon


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def training_step(self, batch, batch_idx):
        # print("batch_idx: ", batch_idx)
        index, img_batch_can, img_batch_ran = batch # Plese check if range of img is [-1.0, 1.0]
        results_dict     = self.model.forward(img_batch_ran)
        loss             = self.model.loss_function(
            **results_dict,
            x_can        = img_batch_can,
            x_ran        = img_batch_ran,
            loss_weight  = self.loss_weight,
            batch_idx    = batch_idx
        )
        # self.save_progress(img_batch, results_dict)
        self.log("val_loss", loss["loss"])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)
        return loss['loss']


    def validation_step(self, batch, batch_idx):
        index, img_batch_can, img_batch_ran = batch  # shape = [num_batch, step, channel, w, h], Eg.) [128, 8, 3, 64, 64])
        results_dict     = self.model.forward(img_batch_ran)
        loss             = self.model.loss_function(
            **results_dict,
            x_can        = img_batch_can,
            x_ran        = img_batch_ran,
            loss_weight  = self.loss_weight,
            batch_idx    = batch_idx
        )
        self.log("val_loss", loss["loss"])
        if batch_idx == 0:
            self.save_progress(img_batch_can, results_dict, name_tag="_canonical")
            self.save_progress(img_batch_ran, results_dict, name_tag="_randomized")


    def save_progress(self, img_batch, results_dict: dict, name_tag: str=""):
        if pathlib.Path(self.logger.log_dir).exists():
            p = pathlib.Path(self.logger.log_dir + "/reconstruction"); p.mkdir(parents=True, exist_ok=True)
            num_batch, step, channel, width, height = img_batch.shape


            save_sequence = 8  # np.minimum(10, mod)
            images        = []
            # permute       = [2, 1, 0] # BGR --> RGB for accurate save using PIL
            for n in range(save_sequence):
                images.append(utils.make_grid(results_dict["x_recon"][n], nrow=step, normalize=True))
                images.append(utils.make_grid(              img_batch[n], nrow=step, normalize=True))

            print("\n\n---------------------------------------")
            print(" [img_batch] min. max = [{}, {}]".format(img_batch[1].min(), img_batch[1].max()))
            print(" [  images ] min. max = [{}, {}]".format(   images[1].min(),    images[1].max()))
            print("---------------------------------------\n\n")

            # save input and reconstructed images
            '''
                Plese check if range of img is [0.0, 1.0].
                Because utils.save_image() assums that tensor image is in range [0.0, 1.0] internally.
            '''
            utils.save_image(
                tensor = torch.cat(images, dim=1),
                fp     = os.path.join(str(p), 'reconstruction_epoch' + str(self.current_epoch)) + name_tag + '.png',
            )
