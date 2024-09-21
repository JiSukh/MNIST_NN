import numpy as np
import data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax






inputs = [-22,10]




layer1 = DenseLayer(2,3)

layer1.forward(inputs)

activate = ActivationReLU()

activate.forward(layer1.output)

activateSoftMax = ActivationSoftMax()

activateSoftMax.forward(layer1.output)


print(f"Outputs: {layer1.output}")
print(f"Activation ReLU: {activate.output}")
print(f"Activation SoftMax: {activateSoftMax.output}")
