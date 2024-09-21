import numpy as np


class ActivationSoftMax:
    """
    Activate a stable soft max algorithm to an entire layer
    """
    def forward(self, inputs):
        z = inputs - max(inputs)
        numerator = np.exp(z)
        denominator = np.sum(numerator)

        self.output = numerator/denominator