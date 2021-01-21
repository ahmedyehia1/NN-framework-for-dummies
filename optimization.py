# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:16:28 2021

@author: Memad
"""

import numpy as np

#output of activation function is model.layers[n].output [1 x number of neurons]
#loss function used is model.loss (string)
#input example is model.x vector[1 x n] (row per example)
#label vector model.y (value for each example)
#drevative of activation function of each layer===> model.layers[n].Adash
#NOTES:
    #EPSILON OF THE MODEL IS NEEDED
    #ALPHA OF THE MODEL IS NEEDED
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
    
def sgd(model,alpha,sample):
    for i in range(len(model.layers)):
        n = len(model.layers) - i -1
        if(i == 0): #output layer ==> there is no saved drivative value
            dloss_dact = dl_dy(model.loss,model.layers[n].output,model.y,model.x,model.label)            
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
        model.layers[i].weights=model.layers[n].weights-alpha*dl_dw
        model.layers[i].bias=model.layers[n].bias-alpha*np.transpose(dl_db)
        
        
def batch(model,sample):
    for i in range(len(model.layers)):
        n = len(model.layers) - i -1
        if(i == 0): #output layer ==> there is no saved drivative value
            dloss_dact = dl_dy(model.loss,model.layers[n].output,model.y,model.x,model.label)            
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
