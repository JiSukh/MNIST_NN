# Simple Neural Network for MNIST Dataset

This is a simple neural network created to work with the MNIST dataset. Not many advanced things but most concepts of deep learning is here.


## Overview

This project demonstrates a basic neural network designed for digit classification on the MNIST dataset. The project uses nothing but **numpy** for the neural network. To save the network, pickle in a kinda hacky way but the functionality is all there.

Additionally, openCV is used only to rescale and prepare new image data before it gets inputted into the network.

## Methodologies
- Basic implementation of dense layers
- ReLU and Stable Soft Max activation functions
- Cross Entropy Loss Function
- Simple clipping to prevent gradient explosion
- Basic gradient descent optimiser

Disclaimer: This project does not implement many of the functions and methods that helps tune more advanced neural networks, like dropouts, advanced optimisers like Adam, or batch processing. 

This is a simple network with an input layer, 2 hidden layers, and an output layer.

## Requirements

- Python 3
- NumPy
- Matplotlib

You can install all dependencies using the following command:

```bash
pip install -r requirements.txt
