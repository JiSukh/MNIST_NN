import numpy as np
import data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax
from Cost import CrossEntropy






inputs = [-22,10]

ytrue = [1,0,0]

ypred_testing = [0.96,0.3,0.1]





layer1 = DenseLayer(2,3)

layer1.forward(inputs)

activate = ActivationReLU()

activate.forward(layer1.output)

activateSoftMax = ActivationSoftMax()

activateSoftMax.forward(layer1.output)

crossEntropy = CrossEntropy()

ypred = activateSoftMax.output

crossEntropy.forward(ypred_testing, ytrue)

print(f"Outputs: {layer1.output}")
print(f"Activation ReLU: {activate.output}")
print(f"Activation SoftMax: {activateSoftMax.output}")

print(f"Loss: {crossEntropy.output}")
