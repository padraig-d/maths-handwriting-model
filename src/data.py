from torch import float32
from torch.utils.data import random_split

from torchvision.transforms import v2 as transforms
from torchvision import datasets

# TODO turn these into environment variables, or better yet parameters
DATA_ROOT = "data/"
TRAINING_PERCENT = 0.7

# TODO finalise the transformations we'll be using
transform = transforms.Compose((
    # convert the PIL image into a tensor
    transforms.ToImage(),
    transforms.ToDtype(float32, scale=True),

    transforms.Grayscale(), # drop the unneeded dimensions
    # transforms.RandomResizedCrop(), # find out how to use this
    # normalising images is done on photos of the real world; on handwriting this is not as necessary
    # transforms.Normalize((0.5,), (0.5,)),
))

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform) # the entire dataset, preprocessed

train_data, test_data = random_split(dataset, [TRAINING_PERCENT, 1 - TRAINING_PERCENT])
