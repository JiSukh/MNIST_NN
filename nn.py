import numpy as np
from data import get_data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax
from Cost import CrossEntropy


images, labels = get_data()

epochs = 10

#create new neural network

inputlayer = DenseLayer(784,12)
hiddenlayer1 = DenseLayer(12,12)
hiddenlayer2 = DenseLayer(12,10)
inputReLU = ActivationReLU()
hiddenlayer1ReLU = ActivationReLU()
outputSoftMax = ActivationSoftMax()
CrossE = CrossEntropy()

#forward pass
inputlayer.forward(images[0])
inputReLU.forward(inputlayer.output)

hiddenlayer1.forward(inputReLU.output)
hiddenlayer1ReLU.forward(hiddenlayer1.output)

hiddenlayer2.forward(hiddenlayer1ReLU.output)
outputSoftMax.forward(hiddenlayer2.output)

CrossE.forward(outputSoftMax.output, labels[0])

print(CrossE.output)
