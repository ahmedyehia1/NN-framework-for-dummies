import numpy as np
import optimization as opt
from Activations import *
import Losses as loss
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

        # check if the size forms of a tuple and activation as a string
        assert(type(size) == tuple)
        assert(type(activation) == str)

        self.inSize = size[0]
        self.outSize = size[1]
        # initialize weights and biases
        # weight of size (layer neurans no. , prev layer neurans no.)
        self.weights = np.random.randn(self.outSize,self.inSize)
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

        Return:
            a forward propagation step value after applying activation function

        Shape:
            - Input: inputs (1, N) N: number of features
            - Output: (1, m)

        Examples:
            layer = nn.Layer(size=(3,5), activation='ReLU')
            result = layer.forward(np.array([4,1,-5,3]))
        '''
        
        assert(type(inputs) == np.ndarray)
        assert(inputs.shape[1] == self.inSize)

        # linear w*x + b forward propagtion step
        self.Z = np.dot(inputs,self.weights.T) + self.bias

        # evaluate the results after the activation
        self.A, self.Adash = eval(self.activation+"(Z = self.Z)")
        return self.A
        
    def __call__(self,inputs):
        '''
        Calculate a forward propagation step over one layer:
        based on the Linear formula:
            Y = activation(X.W + b)

        Args:
            inputs: a numpy array with the previous layer values (or the network inputs)

        Return:
            a forward propagation step value after applying activation function

        Shape:
            - Input: inputs (1, N) N: number of features
            - Output: (1, m)

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

        Return:
            a forward propagation iteration value for the whole network
        
        Shape:
            - Input: inputs (1, N) N: number of features
            - Output: (1, m) m: number of nurans at output

        Examples:
            model = nn.Model(
                Layer(size=(3,5), activation='ReLU'),
                Layer(size=(5,1), activation='ReLU')
            )
            model.forward(np.array([1,6,-2]))
        '''
        
        assert(type(inputs) == np.ndarray)

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

        Return:
            a forward propagation iteration value for the whole network
        
        Shape:
            - Input:  inputs (1, N) N: number of features
            - Output: (1, m) m: number of nurans at output

        Examples:
            model = nn.Model(
                Layer(size=(3,5), activation='ReLU'),
                Layer(size=(5,1), activation='ReLU')
            )
            model(np.array([1,6,-2]))
        '''

        return self.forward(inputs)



    def fit(self,dataset_input,label,optimization_type,loss_type,alpha,epsilon):
        if(optimization_type == 'SGD'):
            while(True):
                opt.init_delta(self)
                for i in range(len(dataset_input)):
                    self.forward(dataset_input[i].reshape(1,-1))
                    loss_value , dloss = loss.mse_loss(self.y,label[i].reshape(1,-1))
                    opt.sgd(self,alpha,dataset_input[i].reshape(1,-1),dloss)                    
                if(opt.norm(self) < epsilon):
                    break
          #-------------------------------------------      
        elif(optimization_type == 'batch'):
              while(True):
                opt.init_delta(self)
                for i in dataset_input:
                    self.forward(dataset_input[i])
                    loss_value , dloss = loss.mse_loss(self.y,label[i])
                    opt.batch(self,alpha,dataset_input[i],dloss)
                opt.update_weights_bias(self,alpha,len(dataset_input)) 
                if(opt.norm(self,len(dataset_input)) < epsilon):
                    break



    def evaluate(self,test_x,test_y,metric='Accuracy',beta=1.0):
        '''
        Calculate the evaluation matrices for the testing dataset

        Args:
            test_x: a numpy array with the network testing inputs
            test_y: a numpy array with the network testing true labels
            metric: metric name(s) as string or list of strings:
                    avliable metrics:
                        - Accuracy
                        - Confusion matrix
                        - Precision
                        - Recall
                        - F1 score
                        - FBeta score
                Default: Accuracy
            beta: a hyperparameter value used to calculate FBeta score

        Shape:
            - Input:
                test_x (K, N) K:number of testing samples, N: number of features
                test_y (K, m) K:number of testing samples, m: number of nurans at output
            - Output:
                the specified metric value(s)
                beside storing all metrics in model variables as follows:
                    self.accuracy [0-1] value
                    self.confusion_matrix (2,2) matrix
                    self.recall  [0-1] value
                    self.precision [0-1] value
                    self.f1_score [0-1] value
                    self.fbeta_score [0-1] value
        Examples:
            P,R,F1 = model.evaluate(
                        np.array([[1,6,-2],[3,9,12],[7,-3,4]]),
                        np.array([[0],[1],[1]]),
                        metric=['Precision','Recall','F1_score','FBeta_score'],
                        beta=0.6
                    )
        '''
        assert(type(test_x) == np.ndarray)
        assert(type(test_y) == np.ndarray)
        assert(type(metric) == str or type(metric) == list)
        assert(type(beta) == float)

        # propagate in forward iteration over testing data
        test_y_hat = self.forward(test_x)

        # calculate True Positive, True Negative, False Positive, False Negative respectively
        self.TP = (np.equal(test_y_hat,1) & np.equal(test_y,1)).sum()
        self.TN = (np.equal(test_y_hat,0) & np.equal(test_y,0)).sum()
        self.FP = (np.equal(test_y_hat,1) & np.equal(test_y,0)).sum()
        self.FN = (np.equal(test_y_hat,0) & np.equal(test_y,1)).sum()

        # calculate evaluation matrics
        self.accuracy = (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN+(10E-9 if self.TP == 0 else 0))
        self.confusion_matrix = np.array([[self.TP,self.FP],[self.FN,self.TN]])
        self.precision = (self.TP)/(self.TP+self.FP+(10E-9 if self.TP+self.FP == 0 else 0))
        self.recall = (self.TP)/(self.TP+self.FN+(10E-9 if self.TP+self.FN == 0 else 0))
        self.f1_score = 2*(self.precision*self.recall)/(self.precision+self.recall+(10E-9 if self.TP == 0 else 0))
        self.fbeta_score = (1+beta**2)*(self.precision*self.recall)/(beta**2*self.precision+self.recall+(10E-9 if self.TP == 0 else 0))
        
        # return the targeted metric(s) 
        if type(metric) == str:
            metric = [metric]
        out = list()
        for m in metric:
            out.append(eval('self.'+'_'.join(list(filter(lambda w: w != "" , str.lower(m).split(" "))))))
        return out