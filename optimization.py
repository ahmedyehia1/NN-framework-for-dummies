# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:16:28 2021

@author: Memad
"""
import Losses as loss
import numpy as np
def dl_dy (loss,ycap,y,inp,label):
    if(loss == 'mse'):#Regression
        return(ycap-y)
# =============================================================================
#     if(loss == 'svm'):#Classification
#         delta = np.zeros((len(ycap),len(inp)))
#         d = np.maximum(1 + ycap - ycap[label], 0)
#         d[label] = 0
#         d = d > 0
#         delta[d] = inp
#         delta[label] = -inp * np.sum(d)
#         return delta 
# =============================================================================
    
def sgd(model,alpha,sample,dloss):
    for i in range(len(model.layers)):
        n = len(model.layers) - i -1
        if(i == 0): #output layer ==> there is no saved drivative value
            dloss_dact =dloss             
            dl_dw=np.dot(np.multiply(np.transpose(dloss_dact) ,np.transpose(model.layers[n].Adash)),model.layers[n-1].A)
            model.layers[n].weights_Grad += dl_dw
            dl_db=np.multiply(np.transpose(dloss_dact) ,np.transpose(model.layers[n].Adash))
            model.layers[n].bias_Grad += np.transpose(dl_db)
            saved_drevative = np.dot(model.layers[n].weights.transpose(),dl_db)
        else:       #hidden layers
            dactivation = np.multiply(saved_drevative ,np.transpose(model.layers[n].Adash))
            if(n == 0):
                dl_dw=np.dot(dactivation,sample)
            else:
                dl_dw=np.dot(dactivation,model.layers[n-1].A)
            model.layers[n].weights_Grad += dl_dw
            dl_db=dactivation
            model.layers[n].bias_Grad += np.transpose(dl_db)
            print(np.shape(model.layers[n].weights))        
            saved_drevative = np.dot(model.layers[n].weights.transpose(),saved_drevative)        
        model.layers[i].weights=model.layers[n].weights-alpha*dl_dw
        model.layers[i].bias=model.layers[n].bias-alpha*np.transpose(dl_db)
        print(model.layers[i].weights)
        
def batch(model,sample,dloss):
    for i in range(len(model.layers)):
        n = len(model.layers) - i -1
        if(i == 0): #output layer ==> there is no saved drivative value
            dloss_dact = dloss            
            dl_dw=np.dot(np.multiply(np.transpose(dloss_dact) ,np.transpose(model.layers[n].Adash)),model.layers[n-1].A)
            model.layers[n].weights_Grad += dl_dw
            dl_db=np.multiply(np.transpose(dloss_dact) ,np.transpose(model.layers[n].Adash))
            model.layers[n].bias_Grad += np.transpose(dl_db)
            saved_drevative = np.dot(model.layers[n].w.transpose(),dl_db)
        else:       #hidden layers
            dactivation = np.multiply(saved_drevative ,np.transpose(model.layers[n].Adash))
            if(n == 0):
                dl_dw=np.dot(dactivation,sample)
            else:
                dl_dw=np.dot(dactivation,model.layers[n-1].A)
            model.layers[n].weights_Grad += dl_dw
            dl_db=dactivation
            model.layers[n].bias_Grad += np.transpose(dl_db)             
            saved_drevative = np.dot(model.layers[n].weights.transpose(),saved_drevative)


def norm(model, size_of_dataset):
    norms_weights = 0.0
    norms_bias = 0.0
    for i in range(len(model.layers)):
        norms_weights += np.norm(model.layers[i].weights_Grad / size_of_dataset)
        norms_bias += np.norm(model.layers[i].bias_Grad / size_of_dataset)
    return norms_weights + norms_bias


def init_delta(model):
    for i in range(len(model.layers)):
        model.layers[i].weights_Grad = np.zeros_like(model.layers[i].weights)
        model.layers[i].bias_Grad = np.zeros_like(model.layers[i].bias)


def update_weights_bias(model, alpha, size_of_dataset):
    for i in range(len(model.layers)):
        model.layers[i].weights = model.layers[i].weights - alpha * (model.layers[i].weights_Grad / size_of_dataset)
        model.layers[i].bias = model.layers[i].bias - alpha * (model.layers[i].bias_Grad / size_of_dataset)
