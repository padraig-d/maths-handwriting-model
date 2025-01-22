# progrma to run matplotlib to check dataset

import matplotlib.pyplot as plt
from data import split_dataset, load_data

train_data, test_data = split_dataset()
trainloader, testloader = load_data(train_data, test_data)

train_features, train_labels = next(iter(trainloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0] 
label = train_labels[0] 

plt.imshow(img.permute(1, -1, 0)) # this is necessary to transpose the size shape from (1, 28, 28) to (28, 28, 1)
plt.show()