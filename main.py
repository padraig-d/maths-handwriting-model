from src import LetNet
from torchvision import models
from scripts.runner import run
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


LetModel = models.alexnet() # LetNet.Net()
LetModel.classifier[6] = torch.nn.Linear(in_features=4096, out_features=18)

run(LetModel)