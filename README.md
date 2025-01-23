## Getting Started
### Python
PyTorch supports Python 3.9 - 3.12.
Check that your version falls into that range with `python -V`. If it doesn't, I recommend you install `python3.12` and create a virtual environment.
```sh
sudo apt install python3.12 -y # your distribution's package manager might not be apt
python3.12 -m venv .venv # create a virtual environment and put it in the hidden folder .venv
```
In order to use that virtual environment, you need to run the activation script for every new shell session.
This is because the script temporarily changes your version of Python.
```sh
source .venv/bin/activate
```
After this, you can continue.

### Dependencies
Standard dependencies are in `requirements.txt`.
```sh
pip install -r requirements.txt
```

PyTorch is not multiplatform, so you'll need to select the correct version for your hardware [here](https://pytorch.org/get-started/locally/). Follow the instructions and run the command given. If you have no GPU, pick the CPU option.

## Data
Dataset consists of 27,000 images of the symbols `0,1,2,3,4,5,6,7,8,9,*,-,+,/,w,x,y,z` (1,500 each).
It has yet to be sorted correctly.

## USAGE

Run main.py to train the model, it will be saved into /trained_model.

To run various tests on it (such as testing on images in /testing_images), run testing.py