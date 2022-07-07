import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from MNIST_data_setup import MNIST_data_setup
from model.ModelFactory import ModelFactory
from dataloader.DataLoaderFactory import DataLoaderFactory
from optimizer.OptimizerFactory import OptimizerFactory
from loss_function.LossFunctionFactory import LossFunctionFactory

print(torch.cuda.is_available())


class Train:
    def run(self, config):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # dev = torch.device("cpu")

        mnist = MNIST_data_setup()
        mnist.download()
        ((x_train, y_train), (x_valid, y_valid), _) = mnist.load()

        DataLoader = DataLoaderFactory().create(config.dataloader.name); print(DataLoader)
        train_dl   = DataLoader((x_train, y_train), config=config.dataloader, dev=dev)
        valid_dl   = DataLoader((x_valid, y_valid), config=config.dataloader, dev=dev)

        model      = ModelFactory().create(**config.model)
        model.to(dev)

        loss_func  = LossFunctionFactory().create(config.loss_function)
        opt        = OptimizerFactory().create(model.parameters(), **config.optimizer)
        print(model); print(loss_func); print(opt)

        writer         = SummaryWriter('runs/MNIST')
        dataiter       = iter(valid_dl)
        images, labels = dataiter.__next__()
        img_grid       = torchvision.utils.make_grid(images)
        writer.add_image('MNIST_valid_dl', img_grid)
        writer.add_graph(model, images)

        for epoch in range(config.epoch):
            start_time = time.time()
            model.train()
            for xb, yb in train_dl:
                pred = model(xb)
                loss = loss_func(pred, yb) # <-どういう引数を渡すかはモデル依存なので汎化性がない

                loss.backward()
                opt.step()
                opt.zero_grad()

            model.eval()
            with torch.no_grad():
                valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

            print(epoch, valid_loss, time.time() - start_time)
            writer.add_scalar('training loss', loss, epoch)
            writer.add_scalar('validation loss', valid_loss, epoch)

        writer.close()


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def get_config(cfg: DictConfig) -> None:
        train = Train()
        train.run(cfg)

    get_config()