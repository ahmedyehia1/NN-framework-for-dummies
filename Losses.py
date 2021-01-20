import numpy as np

def mse_loss(A,Y):
    m = A.shape[0]
    Loss = (1/(2*m))*np.sum(np.power(A-Y,2))
    Loss_dash = np.multiply((1/m),A-Y)
    return Loss,Loss_dash

def NLL_loss(A,Y):
    Loss  = np.sum(-np.log(np.sum(np.multiply(A,Y),axis = 1,keepdims=True)))
    Loss_dash = np.divide(-1,Y)
    return Loss, Loss_dash

def L1_loss(A,Y):
    m = A.shape[0]
    Loss = (1/(m))*np.sum(np.abs(A-Y))
    vf =  np.vectorize(lambda x: 1 if x > 0 else -1)
    Loss_dash = (1/(m))*vf(A-Y)
    return Loss,Loss_dash



