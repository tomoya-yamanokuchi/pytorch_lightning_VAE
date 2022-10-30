import torch
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.datamodule.DataModuleFactory import DataModuleFactory
from torchvision.io import write_video
from custom.utility.image_converter import torch2numpy

name     = "sprite"
data_dir = "./data/"

datamodule = DataModuleFactory().create(name, data_dir)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()

iter_dataloader = iter(dataloader)
index, batch,   = next(iter_dataloader)
assert index[0] == 0

index, batch,   = next(iter_dataloader)
index, batch,   = next(iter_dataloader)

num_batch, step, channel, width, height = batch.shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import ipdb; ipdb.set_trace()

for i in range(num_batch):
    img_seq = batch[i]
    img_seq = torch2numpy(img_seq)
    write_video(
        filename    = "/home/tomoya-y/workspace/pytorch_lightning_VAE/fig/{}_num_batch{}.mp4".format(name, i),
        video_array = img_seq,
        fps         = 10.0,
    )

