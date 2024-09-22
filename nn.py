import numpy as np
from data import get_data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax
from Cost import CrossEntropy
import cv2
import matplotlib.pyplot as plt
from network_handler import NeuralNetwork


images, labels = get_data()

nn = NeuralNetwork()

#nn.train(images,labels)

while(True):
    nn.run(input('Input path: '))