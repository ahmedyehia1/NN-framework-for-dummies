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
                        -Tanh 
                        -Softmax
                    Default: Identity

        Shape:
            - Input: size (N, m) N: number of features
            - Output: None

        Examples:
            layer1 = nn.Layer(size=(3,5), activation='ReLU')

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

        # initialize A, Adash and Z where
        # Z:        linear output of layer
        # A:        layer output after activation 
        # Adash:   derivative of the layer output after activation
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
            - Input: size (1, N) N: number of features
            - Output: size(1, m)

        Examples:
            layer = nn.Layer(size=(3,5), activation='ReLU')
            result = layer.forward(np.array([4,1,-5,3]))
        '''
        
        assert(type(inputs) == np.ndarray)
        assert(inputs.shape[1] == self.inSize)

        # linear w*x + b forward propagtion step
        self.Z = np.dot(inputs,self.weights) + self.bias

        # evaluate the results after the activation
        self.A, self.Adash = eval(self.activation+"(value = self.Z, inputs = inputs)")
        return self.A
        
    def __call__(self,inputs):
        '''
        Calculate a forward propagation step over one layer:
        based on the Linear formula:
            Y = activation(X.W + b)

        Args:
            inputs: a numpy array with the previous layer values (or the network inputs)

        Shape:
            - Input: size (1, N) N: number of features
            - Output: size(1, m)

        Examples:
            layer = nn.Layer(size=(3,5), activation='ReLU')
            result = layer(np.array([4,1,-5,3]))

        '''
        return self.forward(inputs)


# model class encapsulates Layers objects
class Model:
    def __init__(self,*layers):
        '''
        Create a Model with a specific number and type of Layers:

        Args:
            layers: multi-valued parameter that holds one or more Layer object

        Shape:
            - layers: (nn.Layer(N,m), nn.Layer(m,k), .... , nn.Layer(c,1))
            - Output: None

        Examples:
            model = nn.Model(
                Layer(size=(3,5), activation='ReLU'),
                Layer(size=(5,10), activation='ReLU'),
                Layer(size=(10,6), activation='ReLU'),
                Layer(size=(6,1), activation='ReLU')
                )

        '''

        assert(type(layers) == tuple)
        assert(type(layer) == "nn.Layer" for layer in layers)
        self.x = None
        self.y = None
        self.layers = list(layers)
    
    def forward(self,inputs):
        '''
        Calculate a forward propagation step over the whole model:
        based on the Linear formula:
            A = activation(Layer(x))
            B = activation(Layer(A))
            .
            .
            .
            Y = activation(Layer(F))

        Args:
            inputs: a numpy array with the network inputs

        Shape:
            - Input: size (1, N) N: number of features
            - Output: size(1, m) m: number of nurans at output

        Examples:
            model = nn.Model(
                Layer(size=(3,5), activation='ReLU'),
                Layer(size=(5,1), activation='ReLU')
            )
            model.forward(np.array([1,6,-2]))
        '''

        self.x = inputs
        for layer in self.layers:
            inputs = layer.forward(inputs)
        self.y = inputs
        return self.y

    def __call__(self,inputs):
        '''
        Calculate a forward propagation step over the whole model:
        based on the Linear formula:
            A = activation(Layer(x))
            B = activation(Layer(A))
            .
            .
            .
            Y = activation(Layer(F))

        Args:
            inputs: a numpy array with the network inputs

        Shape:
            - Input: size (1, N) N: number of features
            - Output: size(1, m) m: number of nurans at output

        Examples:
            model = nn.Model(
                Layer(size=(3,5), activation='ReLU'),
                Layer(size=(5,1), activation='ReLU')
            )
            model(np.array([1,6,-2]))
        '''

        return self.forward(inputs)

    def fit(self,epoch,loss_fn='mse',optim='sgd',optim_options={'lr': 0.001}):
        ## to be added
        self.loss = loss_fn
        pass