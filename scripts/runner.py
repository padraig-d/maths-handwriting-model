# code to run the code
from src.data import split_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data import split_dataset, LOADER_PARAMS
from src.train import train
from src.test import test

import torch.nn as nn
from torch.utils.data import DataLoader

# loading in the data, requires initializing, splitting the data and then loading it in to PyTorch loaders

train_set, test_set = split_dataset()
train_loader = DataLoader(train_set, **LOADER_PARAMS, shuffle=True)
test_loader = DataLoader(test_set, **LOADER_PARAMS, shuffle=False)

print(f"batch size: {LOADER_PARAMS["batch_size"]}")

def run(net, epochs):
    train(net, train_loader, epochs)
    print('Finished Training')
    labels, predictions = test(net, test_loader)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    print(
        f'Accuracy: {accuracy:.4f}\n'
        f'Precision: {precision:.4f}\n'
        f'Recall: {recall:.4f}\n'
        f'F1 Score: {f1:.4f}'
    )
