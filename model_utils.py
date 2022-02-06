import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import FloatTensor
import dill
def init_weights(m, M: Module = torch.nn.Linear, initializer = 'Xavier_uniform'):
    if isinstance(m, M):
        if initializer == 'Xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight)
        elif initializer == 'He_normal':
            torch.nn.init.kaiming_normal_(m.weight)

def train_loop(dataloader: DataLoader, model: Module, loss_fn, optimizer, clip_grad = 0):
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.float())
        loss = loss_fn(pred, y.float())
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
    return train_loss / len(dataloader)

def test_loop(dataloader: DataLoader, model: Module, loss_fn):
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            test_loss += loss_fn(pred, y.float()).item()
    return test_loss / len(dataloader)

def save_model_and_valid(model, dataloaders, prefix: str):
    torch.save(model, f'{prefix}_model.pth', pickle_module=dill)
    torch.save(dataloaders['valid'], f'{prefix}_valid_dataloader.pth', pickle_module=dill)

def load_model_and_valid(prefix: str):
    return torch.load(f'{prefix}_model.pth'), torch.load(f'{prefix}_valid_data.pth')

class Model(Module):
    # def __init__(self, train_dataloader, test_dataloader):
    #     self.train_dataloader = train_dataloader
    #     self.test_dataloader = test_dataloader
    def __init__(self):
        super(Model, self).__init__()

    def train_model(self, train_dataloader, test_dataloader, epochs, d_epochs, loss_fn, optimizer, max_grad = 0):
        train_loss, test_loss = [], []
        counter = 0
        for t in range(epochs):
            train_l = train_loop(train_dataloader, self, loss_fn, optimizer, max_grad)
            test_l = test_loop(test_dataloader, self, loss_fn)
            if (t + 1) % d_epochs == 0:
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loss.append(train_l)
                test_loss.append(test_l)
                print(f"Avg train loss: {train_loss[-1]:.8f}")
                print(f"Avg test loss: {test_loss[-1]:.8f}")
                # simple early stopping
                if len(test_loss) < 2:
                    pass
                elif test_loss[-1] >= test_loss[-2]:
                    counter += 1
                else:
                    counter = max(0, counter - 1)
                if counter >= 3:
                    break

    def evaluate_model(self, valid_dataset, criterion = 'MSE'):
        if criterion == 'MAE' or 'RMSE':
            loss_fn = torch.nn.L1Loss()
        elif criterion == 'MSE':
            loss_fn = torch.nn.MSELoss()

        total_loss = 0
        for X, y in valid_dataset:
            pred = self((self.f(X)).float())
            target = self.g(X,y)
            total_loss = loss_fn(pred, target)

        if criterion == 'RMSE':
            return (total_loss.item() / len(valid_dataset))**(1/2)
        else:
            return total_loss.item() / len(valid_dataset)