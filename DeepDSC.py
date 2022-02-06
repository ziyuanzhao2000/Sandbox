import torch
from torch import nn
from model_utils import Model, init_weights

def id(x):
    return x


def X_extract_gene_expr(l: torch.Tensor):
    return l[1]

def Y_extract_gene_expr(x, y):
    return x[1]

class AutoEncoder(Model):
    def __init__(self, l0: int, l1 = 2000, l2 = 1000, l3 = 500):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(l0, l1),
            nn.SELU(),
            nn.Linear(l1, l2),
            nn.SELU(),
            nn.Linear(l2, l3),
            nn.SELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(l3, l2),
            nn.SELU(),
            nn.Linear(l2, l1),
            nn.SELU(),
            nn.Linear(l1, l0),
            nn.Sigmoid()
        )
        self.encoder.apply(lambda m: init_weights(m, initializer='Xavier_uniform'))
        self.decoder.apply(lambda m: init_weights(m, initializer='Xavier_uniform'))

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


class DeepDSC(Model):
    def __init__(self, l0, l1 = 1000, l2 = 800, l3 = 500, l4 = 100, ae = None):
        super(DeepDSC, self).__init__()
        if ae:
            self.autoencoder = ae
        else:
            # stub! Need to have a way to recursively initialize the model using
            # a summary of the data that we will use to train the model
            self.autoencoder = None
        self.stack = nn.Sequential(
            nn.Linear(l0, l1),
            nn.ELU(),
            nn.Linear(l1, l2),
            nn.ELU(),
            nn.Linear(l2, l3),
            nn.ELU(),
            nn.Linear(l3, l4),
            nn.ELU(),
            nn.Linear(l3, 1)
        )
        self.stack.apply(lambda m: init_weights(m, initializer='He_normal'))

    def forward(self, concat_feature: torch.Tensor):
        return self.stack(concat_feature)
