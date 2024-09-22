import numpy as np


class ActivationSoftMax:
    """
    Activate a stable soft max algorithm to an entire layer
    """
    def forward(self, inputs):
        z = inputs - np.max(inputs, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)

        self.output = numerator/denominator