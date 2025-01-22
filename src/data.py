#%%

import matplotlib.pyplot as plt

# neural net imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import float32, no_grad, max
from torch.utils.data import random_split, Dataset, DataLoader

from torchvision.transforms import v2 as transforms
from torchvision import datasets

# TODO add environment variables for these
DATA_ROOT = "../data/"
TRAINING_PERCENT = 0.7

# TODO finalise the transformations we'll be using
TRANSFORM = transforms.Compose((
    # convert the PIL image into a tensor
    transforms.ToImage(),
    transforms.ToDtype(float32, scale=True),

    # images do not need colour
    transforms.Grayscale(), # converts from 3ch RGB to 1ch grayscale
    # lambda i: squeeze(i, dim=0) # removes the 'vestigial' dimension, leaving behind a 2D matrix

    # resize to the necessary size?

    # data augmentation
    # snap a smaller part of the image and expand it to the original size, discarding the rest
    # the scale is the possible area that crop can have - 90% of the image to 100% of it
    # TODO de-hardcode
    transforms.RandomResizedCrop((28, 28), scale=(0.9, 1), antialias=True),

    # normalising images is done on photos of the real world; on handwriting this is not as necessary
    # transforms.Normalize((0.5,), (0.5,)),
))

# changed to init_dataset as we will load the data using LoadDataset
# this is for batch loading

def init_dataset(path = DATA_ROOT): 
    return datasets.ImageFolder(path, transform=TRANSFORM)

def split_dataset(dataset: Dataset = init_dataset()):
    train_data, test_data = random_split(dataset, [TRAINING_PERCENT, 1 - TRAINING_PERCENT])
    return train_data, test_data

def load_data(train_data : Dataset, test_data : Dataset):
    trainloader = DataLoader(train_data, batch_size=4,
                            shuffle=True, num_workers=2)
    testloader = DataLoader(test_data, batch_size=4,
                            shuffle=True, num_workers=2)
    return trainloader, testloader


train_data, test_data = split_dataset()
trainloader, testloader = load_data(train_data, test_data)

train_features, train_labels = next(iter(trainloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0] 
label = train_labels[0] 

# plt.imshow(img.permute(1, -1, 0)) # this is necessary to transpose the size shape from (1, 28, 28) to (28, 28, 1)
# plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # change this from 3 --> 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # changed from (16 * 5 * 5, 120) --> (16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 18)  # changed from (84, 10) --> (84, 18) (the amount of classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # changed from view(-1, 16 * 5 * 5) --> view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
with no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))