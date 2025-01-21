#%%
def main():
    import matplotlib.pyplot as plt

    # neural net imports
    import torch.nn as nn
    import torch.nn.functional as f
    import torch.optim as optim
    import torch.nn.functional as F # I added this so both f and F will work (I dont want to delte not my code) - Jakub

    from torch import float32
    from torch.utils.data import random_split, Dataset, DataLoader

    from torchvision.transforms import v2 as transforms
    # import torchvision.transforms as transforms#this line of code for testing

    from torchvision import datasets

    # TODO add environment variables for these
    DATA_ROOT = "../data/"
    TRAINING_PERCENT = 0.7

    # TODO finalise the transformations we'll be using
    TRANSFORM = transforms.Compose((
        # convert the PIL image into a tensor
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),

        # images do not need colour
        transforms.Grayscale(),  # converts from 3ch RGB to 1ch grayscale
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

    def init_dataset(path=DATA_ROOT):
        return datasets.ImageFolder(path, transform=TRANSFORM)

    def split_dataset(dataset: Dataset = init_dataset()):
        train_data, test_data = random_split(dataset, [TRAINING_PERCENT, 1 - TRAINING_PERCENT])
        return train_data, test_data

    def load_data(train_data: Dataset, test_data: Dataset):
        trainloader = DataLoader(train_data, batch_size=4,
                                 shuffle=True, num_workers=2)
        testloader = DataLoader(test_data, batch_size=4,
                                shuffle=True, num_workers=2)
        return trainloader, testloader

    train_data, test_data = split_dataset()
    trainloader, testloader = load_data(train_data, test_data)

    train_features, train_labels = next(iter(trainloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0]
    label = train_labels[0]

    img = img.expand(3, -1, -1)
    plt.imshow(img.permute(1, -1, 0))  # this is necessary to transpose the size shape from (1, 28, 28) to (28, 28, 1)
    plt.show()

    # %%

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Updated for 28x28 input
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 18)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)  # Updated for 28x28 input
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    import torch
    device = torch.device("cpu")
    net.to(device)

    num_epochs = 5 #epochs number, I looked it up and above 5 is better.
    for epoch in range(num_epochs):
        running_loss = 0.0  # seting value at 0 initializing basically

        # training....................................(hope it aint take long)
        test = 1000 #this is the number of training data it will upload, not sure what order they are uploaded in tho
        net.train()  # model is set in training mode
        for i, data in enumerate(trainloader, 0):
            if i == test:
                break #stops model after 1001 or 1000 images

            print("begin the training has.....") #this prints every images processed, moving this would fix
            inputs, labels = data


            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        if (i + 1) % 2000 == 0:
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0
            #just prints loss every 2000 images

            # Evaluate on test set
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1} -> Test Accuracy: {100 * correct / total:.2f}%")#fscore calculation

    print("Finished Training!")

if __name__ == '__main__':
    main() #this fixes it, apparently you are supposed to wrap only some of the code
    #like for "so it does no execute when you call it" but I wrapped the whole thing
    #this can be fixed --- Jakub
    #not sure why this fixes it but it does

