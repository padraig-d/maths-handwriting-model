# code to run the code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.data import split_dataset, load_data
from torch import float32, no_grad, max, inference_mode
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# loading in the data, requires initializing, splitting the data and then loading it in to PyTorch loaders

train_set, test_set = split_dataset()
testloader, trainloader = load_data(test_set, train_set)

# our neural network

def run(net):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

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


    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)
    
    print(f'Precision of the network on test images: {precision:.4f}')
    print(f'Recall of the network on test images: {recall:.4f}')
    print(f'F1 Score of the network on test images: {f1:.4f}')

    cm = confusion_matrix(all_labels, all_predicted)
    cm = torch.tensor(cm)
    false_positives = cm.sum(axis=0) - torch.diag(cm) # sum of each column excluding the diagonal to get the false positives for each class
    false_negatives = cm.sum(axis=1) - torch.diag(cm) # sum of each row excluding the diagonal to get the false negatives for each class

    total_false_positives = false_positives.sum().item() # sum up all the false positives
    total_false_negatives = false_negatives.sum().item() # sum up all the false negatives

    print(f'Total False Positives: {total_false_positives}') # Print total false positives
    print(f'Total False Negatives: {total_false_negatives}') # Print total false negatives
    print(cm)

    