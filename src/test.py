from torch import nn, inference_mode
from torch import max as torch_max
from torch.utils.data import DataLoader


def test(net: nn.Module, data_loader: DataLoader):
    correct = 0
    total = 0
    with inference_mode():
        for data in data_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch_max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))
    