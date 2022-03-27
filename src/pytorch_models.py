import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data_utils import TwoSpiralDataSet
from tqdm import tqdm
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

device = "cpu"
        
class OneLayerMLQPwithLinearOut(nn.Module):
    def __init__(self):
        super(OneLayerMLQPwithLinearOut, self).__init__()
        input_size = 2
        hidden_size = 40
        output_size = 1
        self.u = nn.Parameter(torch.randn([input_size, hidden_size], dtype=torch.float32))
        self.v = nn.Parameter(torch.randn([input_size, hidden_size], dtype=torch.float32))
        self.b = nn.Parameter(torch.randn([1, hidden_size], dtype=torch.float32))
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.v)
        nn.init.xavier_normal_(self.b)
        
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, X):
        X2 = torch.mul(X, X)
        output = torch.mm(X2, self.u) + torch.mm(X, self.v) + self.b
        output = torch.nn.Sigmoid()(output)
        output = self.linear(output)
        
        return torch.nn.Sigmoid()(output)


class TwoLayerMLP(nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.linear1 = nn.Linear(2, 40)
        self.linear2 = nn.Linear(40, 1)
    def forward(self, X):
        X = self.linear1(X)
        X = nn.ReLU()(X)
        X = self.linear2(X)
        X = nn.Sigmoid()(X)
        return X

class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super(ThreeLayerMLP, self).__init__()
        self.linear1 = nn.Linear(2, 32)
        self.linear2 = nn.Linear(32,64)
        self.linear3 = nn.Linear(64, 1)
    def forward(self, X):
        X = self.linear1(X)
        X = nn.ReLU()(X)
        X = self.linear2(X)
        X = nn.ReLU()(X)
        X = self.linear3(X)
        X = nn.Sigmoid()(X)
        return X



def load_data(data_path="data"):
    data = TwoSpiralDataSet(data_path=data_path)
    data.load_data()
    train_data = TensorDataset(torch.tensor(data.X_train, dtype=torch.float32), torch.tensor(data.Y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(data.X_test, dtype=torch.float32), torch.tensor(data.Y_test, dtype=torch.float32))
    batch_size = 1
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss, current = loss.item(), batch * len(X)
    # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.unsqueeze(0).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def plot_db(model: nn.Module, save_path: str="figures/Torch", index:str="Torch"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    grid_points = []
    for i in np.linspace(-6,6,100):
        for j in np.linspace(-6, 6, 100):
            grid_points.append(np.expand_dims([i,j], axis=0))
    grid_points = np.concatenate(grid_points, axis=0)
    grid_points = torch.tensor(grid_points, dtype=torch.float32)
    size = grid_points.size(0)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(size):
            X = grid_points[i,:].unsqueeze(0).to(device)
            pred = torch.round(model(X))
            preds.append(pred.item())
        preds = np.expand_dims(np.array(preds), 1)
    grid_points = grid_points.cpu().numpy()

    plt.figure(figsize=(5,5))
    plt.scatter(grid_points[:, 0][preds[:,0]==0], grid_points[:, 1][preds[:,0]==0], c='k')
    plt.scatter(grid_points[:, 0][preds[:,0]==1], grid_points[:, 1][preds[:,0]==1], c='w')
    plt.legend(["label=0", "label=1"])
    plt.title(index+" decision boundary")
    plt.savefig(os.path.join(save_path, index+"_decision_boundary.png"), dpi=100)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OneLayerMLQP")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--visualize_db", action="store_true") 
    args = parser.parse_args()

    if args.model_name == "OneLayerMLQPwithLinearOut":
        model = OneLayerMLQPwithLinearOut()
    elif args.model_name == "TwoLayerMLP":
        model = TwoLayerMLP()
    elif args.model_name == "ThreeLayerMLP":
        model = ThreeLayerMLP()
    else:
        raise ValueError("Invalid model_name!")
    
    train_dataloader, test_dataloader = load_data(args.data_path)
    loss_fn = nn.MSELoss()
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    if args.do_train:
        for i in tqdm(range(args.epochs)):
            train(train_dataloader, model, loss_fn, optimizer)
        if args.do_test:
            test(test_dataloader, model, loss_fn)
        if args.visualize_db:
            plot_db(model, save_path="figures/SingleModel", index="Torch_"+args.model_name)



    

    