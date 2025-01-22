from torch import nn, inference_mode, Tensor
from torch import max as torch_max
from torch.utils.data import DataLoader

import numpy as np

def test(net: nn.Module, data_loader: DataLoader):
    max_size = len(data_loader) * (data_loader.batch_size) # the max number of entries we'll need to store
    truth = np.empty(dtype=np.int64, shape=max_size) # this is the datatype of labels
    predictions = np.empty(dtype=np.int64, shape=max_size)

    size = 0
    with inference_mode():
        for data in data_loader:
            images: Tensor
            labels: Tensor
            images, labels = data
            length = len(labels) # should be the same every time, except the last

            truth[size:size + length] = labels.cpu()
            # truth.put(size, labels.cpu())

            outputs = net(images)
            _, predicted = torch_max(outputs.data, 1)
            predictions[size:size + length] = predicted.cpu()
            # predictions.put(size, predicted.cpu())

            size += length
    truth.resize(size, refcheck=False) # disable ref checking because the debugger triggers the error
    predictions.resize(size, refcheck=False)
    return truth, predictions
