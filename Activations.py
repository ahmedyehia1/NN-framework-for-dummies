import numpy as np

def sigmoid(Z):
    A = np.divide(1,1+np.exp(-Z))
    A_dash = np.multiply(A,1-A)
    return A,A_dash

def tanh(Z):
    A = np.tanh(Z)
    A_dash = 1-np.power(A,2)
    return A,A_dash

def relu(Z):
    A = Z*(Z >=0)
    A_dash = 1.*(Z>=0)
    return A,A_dash

def softmax(Z):
    A = np.exp(Z)
    A = np.divide(A,np.sum(A,axis = 1,keepdims=True))
    A_dash = A
    return A,A_dash

def leakyrelu(Z):
    A = np.where(Z > 0, Z, Z * 0.01)
    A_dash = Z
    A_dash[A_dash >= 0] = 1
    A_dash[A_dash < 0] = 0.01
    return A,A_dash

def identity(Z):
    A = Z
    A_dash = np.ones_like(A)
    return A,A_dash