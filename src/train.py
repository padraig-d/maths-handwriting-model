from torch import nn, optim
from torch.utils.data import DataLoader

def train(net: nn.Module, data_loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):  # loop over the dataset multiple times
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print statistics
            # if i % 100 == 0:
            print(f"[{epoch}, {i}] loss: {loss.item():.3f}")