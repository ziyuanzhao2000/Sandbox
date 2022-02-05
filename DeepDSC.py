import torch
from torch import nn
from model_utils import Model

def id(x):
    return x

class AutoEncoder(Model):
    def __init__(self, l0: int, l1: int = 128, f = id, g = id):
        super(AutoEncoder, self).__init__(f, g)
        self.encoder = nn.Sequential(
            nn.Linear(l0, l1),
            nn.SELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(l1, l0),
            nn.SELU()
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


class DeepDSC(Model):
    def __init__(self):
        super(DeepDSC, self).__init__()

    def forward(self):
        # needs to be implemented
        pass
