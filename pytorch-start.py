#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from the colab, normalises the image and transforms it to a tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root="./data", transform=transform)
datatest = DataLoader(trainset, batch_size=64, shuffle=True)

# code taken from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
train_features, train_labels = next(iter(datatest))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0]
img = img.permute(1, -1, 0) # necessary to get the shape of the tensor from (28, 28, 3) to (3, 28, 28)
label = train_labels[0]
plt.imshow(img)
plt.show()
print(f"Label: {label}")

# testing code
item = trainset[26900]
print(item)

#%%