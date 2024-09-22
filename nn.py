import numpy as np
import data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax
from Cost import CrossEntropy




learning_rate = 0.01

inputs = [.2,.5]

ytrue = [1,0,0]

ypred_testing = [0.8,0.29,0.1]





layer1 = DenseLayer(2,3)

layer1.forward(inputs)


print(f"Weight initial: {layer1.weights}")

activate = ActivationReLU()

activate.forward(layer1.output)

activateSoftMax = ActivationSoftMax()

activateSoftMax.forward(layer1.output)

crossEntropy = CrossEntropy()

ypred = activateSoftMax.output

crossEntropy.forward(ypred_testing, ytrue)

crossEntropy.backward(ypred_testing,ytrue)

layer1.backward(crossEntropy.delta_output, learning_rate)

#print(f"Outputs: {layer1.output}")
#print(f"Activation ReLU: {activate.output}")
#print(f"Activation SoftMax: {activateSoftMax.output}")

#print(f"Loss: {crossEntropy.output}")
print(f"Back prop: {crossEntropy.delta_output}")
print(f"Weight after backprop: {layer1.weights}")
