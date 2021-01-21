# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:21:12 2021

@author: Mohamed Amr
"""

import pandas as pd
import math

class DataPreProcessing:
    
    
    def get_data(path, label_path = "", shuffle = False):
        df = pd.read_csv(path)
        #df.columns = df.columns.map(str.lower)
        if(label_path != ""):
            print(label_path)
            df['label'] = pd.read_csv(label_path)['Label']
        if(shuffle):
            df = df.sample(frac = 1)
        X = df.loc[:, df.columns != 'label'].to_numpy()
        label = df['label'].to_numpy()
        return (X, label)
        
    def normalize(matrix):
        mean = matrix.mean()
        variance = matrix.var()
        return (matrix - mean)/math.sqrt(variance+0.0000001)
        
    def split_data(X, label):
        ratio = int(len(label)*0.75)
        X_train = X[:ratio]
        X_test = X[ratio:]
        label_train = label[:ratio]
        label_test = label[ratio:]
        return (X_train, X_test, label_train, label_test)
    
################################################################################
        
#test cases:
#-----------
    
#data  = DataPreProcessing()
#X     = data.get_X('C:/Users/Mohamed Amr/Documents/Computer & Systems Engineering/4th CSE/First Term/Neural Networks/digit-recognizer/train.csv')
#label = data.get_label('C:/Users/Mohamed Amr/Documents/Computer & Systems Engineering/4th CSE/First Term/Neural Networks/digit-recognizer/train.csv')

X, label = DataPreProcessing.get_data("C:/Users/Mohamed Amr/Documents/Computer & Systems Engineering/4th CSE/First Term/Neural Networks/digit-recognizer/train.csv", shuffle = True)
X_train, X_test, label_train, label_test = DataPreProcessing.split_data(X, label)
print("X_train: {}, Label_train: {}, X_test: {}, Label_test: {}".format(X_train.shape, X_test.shape, label_train.shape, label_test.shape))
print(DataPreProcessing.normalize(X))
