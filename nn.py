import numpy as np
from Activations import *

# encapsulate Layers parameters
class Layer:
    def __init__(self,size,activation='Identity'):
        '''
        Create a Layer with a specific size and activation function:

        Args:
            size: a tuble holds (input size, neuran count) respectively
            activation: specify the applied activation function on layer
                    avliable activations:
                        -Identity
                        -ReLU
                        -Sigmoid 
                    Default: Identity

        Shape:
            - Input: size (N, m)
            - Output: None

        Examples:
            layer1 = Layer(size=(3,5), activation='ReLU')

        '''

        #check if the size forms of a tuple and activation as a string
        assert(type(size) == tuple)
        assert(type(activation) == str)

        self.inSize = size[0]
        self.outSize = size[1]
        # initialize weights and biases
        self.weights = np.random.randn(*size)
        self.bias = np.random.randn(1,self.outSize)

        # initialize weights and biases gradient
        self.weights_Grad = np.zeros_like(self.weights)
        self.bias_Grad = np.zeros_like(self.bias)

        # initialize A, A_dash and Z where
        # Z:        linear output of layer
        # A:        layer output after activation 
        # A_dash:   derivative of the layer output after activation
        self.Z = np.zeros_like(self.bias)
        self.A = np.zeros_like(self.bias)
        self.Adash = np.zeros_like(self.bias)

        self.activation = str.lower(activation)

    def forward(self,inputs):
        '''
        Calculate a forward propagation step:
        based on the Linear formula:
            Y = activation(X.W + b)

        Args:
            inputs: a numpy array with the previous layer values (or the network inputs)

        Shape:
            - Input: size (1, N)
            - Output: size(1, m)

        Examples:
            result = Layer.forward(np.array([4,1,-5,3]))
        '''
        
        assert(type(inputs) == np.ndarray)
        assert(inputs.shape[1] == self.inSize)

        # linear w*x + b forward propagtion step
        self.Z = np.dot(inputs,self.weights) + self.bias

        # evaluate the results after the activation
        self.A, self.A_dash = eval(self.activation+"(value = self.Z, inputs = inputs)")
        return self.A
        
    def __call__(self,inputs):
        '''
        Calculate a forward propagation step:
        based on the Linear formula:
            Y = activation(X.W + b)

        Args:
            inputs: a numpy array with the previous layer values (or the network inputs)

        Shape:
            - Input: size (1, N)
            - Output: size(1, m)

        Examples:
            layer1 = Layer.forward(np.array([4,1,-5,3]))
            result = layer1(np.array([4,1,-5,3]))

        '''
        return self.forward(inputs)
