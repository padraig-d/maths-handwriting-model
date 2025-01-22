# code to run the code
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.data import split_dataset, LOADER_PARAMS
from torch import float32, max, inference_mode
from torch.utils.data import DataLoader

# loading in the data, requires initializing, splitting the data and then loading it in to PyTorch loaders

train_set, test_set = split_dataset()
trainloader = DataLoader(train_set, **LOADER_PARAMS, shuffle=True)
testloader = DataLoader(test_set, **LOADER_PARAMS, shuffle=False)

print(f"batch size: {LOADER_PARAMS["batch_size"]}")

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
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


    correct = 0
    total = 0
    with inference_mode():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))
    