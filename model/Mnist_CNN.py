from torch import nn
import torch.nn.functional as F


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 1
        self.w       = 28
        self.h       = 28
        output_shape = 10 # class

        self.f = nn.Sequential(
            nn.Conv2d(self.channel, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_shape),
            # nn.Softmax(dim=1),
        )


    def forward(self, xb):
        xb = xb.view(-1, self.channel, self.w, self.h)  # ; print(xb.shape)
        xb = self.f(xb)                                 # ; print(xb.shape)
        return xb.view(-1, xb.size(1))