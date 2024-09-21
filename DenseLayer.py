import numpy as np



class DenseLayer:
    """Create a dense layer for neural network
    weights = 2d array of weights.
    baies = vector of weights.
    """
    def __init__(self, x, y):
        """
        Create a  dense layer
        x = number of input neurons
        y = number of output neurons.
        """
        self.weights = np.random.rand(x,y) * 0.01 #Create 2d array with weight per each neuron.
        self.biases = np.zeros(y)
        
    def forward(self, inputs):
        """
        Method for forward prop. for dense layer.

        Args:
            inputs (vector): Input vector for this set of neurons. Dataset must be 
        """
        self.output = np.dot(inputs, self.weights) + self.biases