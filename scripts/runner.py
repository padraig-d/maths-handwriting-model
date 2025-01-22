# code to run the code
from src.data import split_dataset, LOADER_PARAMS
from src.train import train
from src.test import test

import torch.nn as nn
from torch.utils.data import DataLoader

# loading in the data, requires initializing, splitting the data and then loading it in to PyTorch loaders

train_set, test_set = split_dataset()
trainloader = DataLoader(train_set, **LOADER_PARAMS, shuffle=True)
testloader = DataLoader(test_set, **LOADER_PARAMS, shuffle=False)

print(f"batch size: {LOADER_PARAMS["batch_size"]}")

# our neural network

def run(net: nn.Module):
    train(net, trainloader)
    print('Finished Training')
    test(net, testloader)