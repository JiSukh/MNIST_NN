import numpy as np
import data
from DenseLayer import DenseLayer




inputs = [2,4]




layer1 = DenseLayer(2,3)

layer1.forward(inputs)


print(layer1.weights)
print(layer1.output)
