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
        self.weights = np.random.rand(x,y) * 0.001 #Create 2d array with weight per each neuron.
        self.biases = np.zeros(y)
        
    def forward(self, inputs):
        """
        Method for forward prop. for dense layer.

        Args:
            inputs (vector): Input vector for this set of neurons.
        """
        self.inputs = np.array(inputs)
        self.output = np.dot(inputs, self.weights) + self.biases
        
        
    def backward(self, delta_outputs, learning_rate):
        """Backwards prop. for dense layer
        """
        self.weights += -learning_rate * np.clip(np.outer(self.inputs, delta_outputs),-1,1)
        self.biases += -learning_rate * np.clip(delta_outputs,-1,1)
        
        self.delta_output = np.dot(delta_outputs, self.weights.T)
        
        