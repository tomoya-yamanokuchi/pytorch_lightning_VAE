from torch import nn

class LinearUnit(nn.Module):
    def __init__(self, in_dim, out_dim, ):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)