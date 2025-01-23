import torch 
import numpy as np

from torch import float32
from torchvision.transforms import v2 as transforms

from PIL import Image
import torch.nn as nn 
import torch.optim as optim 
from src.LetNet import Net
from src.data import init_dataset
from src.test import test
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os

BATCH_SIZE = 16 # variable testing batch-size

model = Net()
# load the saved parameters from a trained model
model.load_state_dict(torch.load("trained_model/model.pt", weights_only=False))

def testing_mode(model : Net):
    model.eval() # turning the model into inference / testing mode

    dataset = init_dataset("data/verification")
    loaded_data = DataLoader(dataset, batch_size=BATCH_SIZE)

    labels, predictions = test(model, loaded_data)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    print(
        f'Accuracy: {accuracy:.4f}\n'
        f'Precision: {precision:.4f}\n'
        f'Recall: {recall:.4f}\n'
        f'F1 Score: {f1:.4f}'
    )


transform_image = transforms.Compose((
    # convert the PIL image into a Tensor, FIRST Grayscale it, very important!
    # transforms.PILToTensor NOT transforms.ToImage()
    transforms.Grayscale(1),
    transforms.PILToTensor(),
    transforms.ToDtype(float32, scale=True),
    
))
# opens the image and transforms it
def process_image(image_path):
    image = Image.open(image_path)
    image = transform_image(image)

    return image


def classify_image(image_path):
    
    # turn on model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preprocessed_image = process_image(image_path).to(device)
    
    # feed one image to model, predictions returns what the model thinks the image is
    with torch.inference_mode():
        predictions = model(preprocessed_image)
    
    predictions = predictions.cpu().numpy()
    predicted_class_index = np.argmax(predictions)
    

    # a dictionary of the classes, classes are indexed 0-18
    classes = {
        0 : "0",
        1 : "1",
        2 : "2",
        3 : "3",
        4 : "4",
        5 : "5",
        6 : "6",
        7 : "7",
        8 : "8",
        9 : "9",
        10 : "dot",
        11 : "minus",
        12 : "plus",
        13 : "slash",
        14 : "w",
        15 : "x",
        16 : "y",
        17 : "z",
    }


    print(f"file:{image_path}")
    print(f"predicted class:{classes[predicted_class_index]}")
    print("__________________________________")



# runner
for file in os.listdir("testing_images/"):
    classify_image(os.path.join("testing_images/", file))



