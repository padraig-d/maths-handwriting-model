#%%
from src import LetNet, test
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

LetModel = LetNet.Net()
#   model, epoch
run(LetModel, 10)


# %%
