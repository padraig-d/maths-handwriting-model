# code to run the code
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.data import split_dataset, load_data
from LetNet import Net
from torch import float32, no_grad, max, inference_mode

# loading in the data, requires initializing, splitting the data and then loading it in to PyTorch loaders

test_set, train_set = split_dataset()
testloader, trainloader = load_data(test_set, train_set)

# our neural network

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

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
with inference_mode():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))