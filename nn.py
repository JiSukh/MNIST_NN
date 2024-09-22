import numpy as np
from data import get_data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax
from Cost import CrossEntropy


images, labels = get_data()
learning_rate = 0.01

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

#Backwards prop
CrossE.backward(outputSoftMax.output, labels[0])

hiddenlayer2.backward(CrossE.delta_output, learning_rate)

hiddenlayer1ReLU.backward()
hiddenlayer1.backward(hiddenlayer1ReLU.delta_output, learning_rate)

inputReLU.backward()
inputlayer.backward(inputReLU.delta_output, learning_rate)


