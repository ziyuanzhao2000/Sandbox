from unittest import TestCase
from data_utils import get_GDSC1_splits, smiles2morgan
from model_utils import save_model_and_valid, load_model_and_valid
from DeepDSC import AutoEncoder, DeepDSC
import torch
class TestAutoEncoder(TestCase):
    def test_train(self):
        try:
            mini_dataloaders = get_GDSC1_splits(frac=[1e-4,1e-4,1e-4], total_transform=lambda a,b:b,
                                                total_transform_2=lambda a,b,c:b)
            ae = AutoEncoder(l0=17737) # any way to get the statistics here rather than a magic val?
            ae.train_model(mini_dataloaders['train'],
                           mini_dataloaders['test'],
                           epochs=10, d_epochs=1,
                           loss_fn=torch.nn.MSELoss(), #cross entropy seems bad??
                           optimizer=torch.optim.Adamax(ae.parameters(), lr=0.0001),
                           max_grad=1)
            save_model_and_valid(ae, mini_dataloaders, 'test_ae')
        except:
            self.fail()

    def test_valid(self):
        try:
            ae, valid_dataloader = load_model_and_valid('test_ae')
            print(ae.evaluate_model(valid_dataloader, criterion='RMSE'))
            print(ae.evaluate_model(valid_dataloader, criterion='MAE'))
        except:
            self.fail()


class TestDeepDSC(TestCase):
    def test_train(self):
        try:
            ae, _ = load_model_and_valid('test_ae')
            mini_dataloaders = get_GDSC1_splits(frac=[1e-4, 1e-4, 1e-4],
                                                fingerprinting=lambda s: smiles2morgan(s, nBits=256),
                                                transform=ae)
            deepdsc = DeepDSC(l0 = 756, ae = ae)
            deepdsc.train_model(mini_dataloaders['train'],
                            mini_dataloaders['test'],
                            epochs=10, d_epochs=1,
                            loss_fn=torch.nn.MSELoss(),  # cross entropy seems bad??
                            optimizer=torch.optim.Adamax(deepdsc.parameters(), lr=0.0004),
                            max_grad=5)
            save_model_and_valid(deepdsc, mini_dataloaders, 'test_deepdsc')
        except:
            self.fail()

    def test_valid(self):
        pass