from torch import nn, optim
from torch.utils.data import DataLoader

EPOCH = 10

def train(net: nn.Module, data_loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_list = []


    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
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
            running_loss += loss.item()
            if i % 2000 == 1999:
                loss_list.append(running_loss / 2000)
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    with open('losses.txt', 'w') as file:
        file.writelines(f"{item}\n" for item in loss_list)