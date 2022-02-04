import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import FloatTensor

def train_loop(dataloader: DataLoader, model: Module, loss_fn, optimizer, f, g):
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(f(X).float())
        loss = loss_fn(pred, g(X,y).float())
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(dataloader)

def test_loop(dataloader: DataLoader, model: Module, loss_fn, f):
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(f(X).float())
            test_loss += loss_fn(pred, g(X,y).float()).item()
    return test_loss / len(dataloader)

class Model(Module):
    # def __init__(self, train_dataloader, test_dataloader):
    #     self.train_dataloader = train_dataloader
    #     self.test_dataloader = test_dataloader

    def train_model(self, train_dataloader, test_dataloader, epochs, d_epochs, loss_fn, optimizer, f, g):
        train_loss, test_loss = [], []
        for t in range(epochs):
            train_l = train_loop(train_dataloader, self, loss_fn, optimizer, f, g)
            test_l = test_loop(test_dataloader, self, loss_fn, f)
            if (t + 1) % d_epochs == 0:
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loss.append(train_l)
                test_loss.append(test_l)
                print(f"Avg train loss: {train_loss[-1]:.8f}")
                print(f"Avg test loss: {test_loss[-1]:.8f}")
                if test_loss[-1] >= test_loss[-2]:  # simple early stopping
                    counter += 1
                else:
                    counter = max(0, counter - 1)
                if counter >= 3:
                    break

    def evaluate_model(self, valid_dataset, criterion = 'MSE'):
        if criterion == 'MAE':
            loss_fn = torch.nn.L1Loss()
        elif criterion == 'MSE':
            loss_fn = torch.nn.MSELoss()

        total_loss = 0
        pred_list = []
        target_list = []
        for X, y in valid_dataset:
            pred_list.append(self(torch.from_numpy(X).float))
            target_list.append(y)
        return loss_fn(FloatTensor(pred_list),
                       FloatTensor(target_list))