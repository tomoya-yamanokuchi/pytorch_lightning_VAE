import torch


class ContentPrior:
    def __init__(self, content_dim):
        self.content_dim = content_dim

    def mean(self, x):
        return torch.zeros(self.content_dim).type_as(x)

    def logvar(self, x):
        # 後でexp()の処理が入るのでここでは0
        return torch.zeros(self.content_dim).type_as(x)
