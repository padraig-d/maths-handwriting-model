from src import LetNet
from src import MyModel
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


LetModel = LetNet.Net()

run(LetModel)