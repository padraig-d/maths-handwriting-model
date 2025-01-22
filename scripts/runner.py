# code to run the code
import torch
import torch.nn.functional as F
from src.data import split_dataset
from torch import max, inference_mode
from torch import max, inference_mode
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.data import split_dataset, LOADER_PARAMS
from src.train import train
from src.test import test

import torch.nn as nn
from torch.utils.data import DataLoader

# loading in the data, requires initializing, splitting the data and then loading it in to PyTorch loaders
def load_data():
    train_set, test_set = split_dataset()
    testloader = DataLoader(test_set, **LOADER_PARAMS, shuffle=False)
    trainloader = DataLoader(train_set, **LOADER_PARAMS, shuffle=True)
    return trainloader, testloader

print(f"batch size: {LOADER_PARAMS["batch_size"]}")

trainloader, testloader = load_data()

def run(net, testloader=testloader, trainloader=trainloader):
    train(net, testloader)

    print('Finished Training')

    correct = 0
    total = 0

    all_predicted = [] # Initialize list to store all predicted labels
    all_labels = [] # Initialize list to store all true labels
    
    with inference_mode():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))

    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)
    
    print(f'Precision of the network on test images: {precision:.4f}')
    print(f'Recall of the network on test images: {recall:.4f}')
    print(f'F1 Score of the network on test images: {f1:.4f}')