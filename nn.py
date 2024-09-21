import numpy as np
import data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU




inputs = [-22,10]




layer1 = DenseLayer(2,3)

layer1.forward(inputs)

activate = ActivationReLU()

activate.forward(layer1.output)


print(f"Outputs: {layer1.output}")
print(f"Activation: {activate.output}")
