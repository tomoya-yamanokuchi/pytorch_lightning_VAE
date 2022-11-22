import copy
import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.test.TestModel import TestModel
from domain.visualize.vector_heatmap import VectorHeatmap


log     = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122171135]"

# ----------------------------------------------------------------------------------
model   = "C-DSVAE"
log_dir = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/{}/".format(model)
test    = TestModel(
    config_dir  = log_dir + log,
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()
# ----------------------------------------------------------------------------------

vectorHeatmap = VectorHeatmap()
for index, img_tuple in dataloader:
    (img, img_aug_context, img_aug_dynamics) = img_tuple
    f = []
    for test_index in range(len(img)):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img_tuple[0])))

        img_seq        = img[test_index].unsqueeze(dim=0).to(device)
        return_dict    = model(img_seq)
        _f             = return_dict["f_mean"].to("cpu").numpy()
        _, dim_f       = _f.shape
        f.append(copy.deepcopy(_f))

    # import ipdb; ipdb.set_trace()
    vectorHeatmap.pause_show(np.concatenate(f, axis=0), interval=1)

