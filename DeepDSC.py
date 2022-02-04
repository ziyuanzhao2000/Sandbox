import torch
from torch import nn
from model_utils import Model

class AutoEncoder(Model):
    def __init__(self, l0: int, l1: int = 128):
        super(AutoEncoder, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(l0, l1),
            nn.ReLU(),
            nn.Linear(l1, l0),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.stack(x)


class DeepDSC(Model):
    def __init__(self):
        super(DeepDSC, self).__init__()

    def forward(self):
        # needs to be implemented
        pass
