from unittest import TestCase
from data_utils import get_GDSC1_splits
from DeepDSC import AutoEncoder
import torch
import dill
class TestAutoEncoder(TestCase):
    def test_train(self):
        try:
            def f(l):
                return l[1]
            def g(x, y):
                return x[1]
            mini_dataloaders = get_GDSC1_splits(frac=[1e-3,1e-3,1e-3])
            ae = AutoEncoder(l0=17737, f=f, g=g) # any way to get the statistics here rather than a magic val?
            ae.train_model(mini_dataloaders['train'],
                           mini_dataloaders['test'],
                           epochs=100, d_epochs=1,
                           loss_fn=torch.nn.MSELoss(),
                           optimizer=torch.optim.Adam(ae.parameters(), lr=0.0001))
            torch.save(ae, 'test_ae.pth', pickle_module=dill)
            torch.save(mini_dataloaders['valid'], 'valid_dataloader.pth')
            print("Test passed!")
        except:
            self.fail()

    def test_valid(self):
        try:
            ae: AutoEncoder = torch.load('test_ae.pth')
            valid_dataloader = torch.load('valid_dataloader.pth')
            print(ae.evaluate_model(valid_dataloader, criterion='MSE'))
            print(ae.evaluate_model(valid_dataloader, criterion='MAE'))
        except:
            self.fail()
