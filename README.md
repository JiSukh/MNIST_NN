# Simple Neural Network for MNIST Dataset

This is a simple neural network created to work with the MNIST dataset with **no deep-learning libraries like tensorflow or pytorch.**


## Overview

This project demonstrates a basic neural network designed for digit classification on the MNIST dataset. 

The project uses nothing but **numpy** for the neural network. 

To save the network, pickle in a kinda hacky way but the functionality is all there.
Lastly, openCV is used only to rescale and prepare new image data before it gets inputted into the network.

## Methodologies
- Basic implementation of dense layers
- ReLU and Stable Soft Max activation functions
- Cross Entropy Loss Function
- Simple clipping to prevent gradient explosion
- Basic gradient descent optimiser

Disclaimer: This project does not implement many of the functions and methods that helps tune more advanced neural networks, like dropouts, advanced optimisers like Adam, or batch processing. 

This is a simple network with an input layer, 2 hidden layers, and an output layer.

This network *should* be able to be used on different datasets as well, given that some configurations about the layers are changed - do so at your own disgression!

## Requirements

- Python 3
- NumPy
- Matplotlib

You can install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Running the project

Navigate to the folder in terminal and run
```bash
python nn.py
```

From there, you can insert new test images into the ```test_images``` folder. The project comes with a model, you can further train with the model using the ```train()``` method in theNeuralNetwork class; change the number of epochs, learning rate, amount of accuracy verification data, etc from there!
