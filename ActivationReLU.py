import numpy as np



class ActivationReLU:
    """
    Take a layer of neurons and activate it using ReLU function.
    
    Args:
        output = Output for ReLU function forward pass
    """
    def forward(self, inputs):
        """Generate forward pass for ReLU 

        Args:
            inputs (vectors): Vector to apply ReLU
        """
        self.output = inputs * (inputs > 0)
    