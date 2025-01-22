from torch import float32
from torch.utils.data import random_split, Dataset, DataLoader

from torchvision.transforms import v2 as transforms
from torchvision import datasets

# TODO add environment variables for these
DATA_ROOT = "data/"
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
    trainloader = DataLoader(train_data, batch_size=32,
                            shuffle=True, num_workers=2)
    testloader = DataLoader(test_data, batch_size=32,
                            shuffle=True, num_workers=2)
    return trainloader, testloader





