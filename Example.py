import numpy as np
import nn
from DataPreProcessing import DataPreProcessing as Data

model = nn.Model(
     nn.Layer(size=(4,5), activation='Relu'),
     nn.Layer(size=(5,3), activation='Relu'),
     nn.Layer(size=(3,10), activation='sigmoid'),
     nn.Layer(size=(10,6), activation='ReLU'),
     nn.Layer(size=(6,1), activation='ReLU')

)

x,label = Data.get_data("data_banknote_authentication.csv")
X_train, X_test, label_train, label_test = Data.split_data(x,label)

# label_train = label_train.reshape(-1,1)
# print(X_train.shape, label_train.shape)

# x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])

# label = np.array([[0], [1], [1], [1]])

'''
x = np.array([
    [2,5,2],
    [5,3,1],
    [4,2,6]
]
)'''


#print(model.forward(x))
model.fit(X_train,label_train,'SGD','MSE',alpha = 0.0001,epoch = 50,graph_on = True)
# print(model.layers[0].Z)
# for i in x:
#     model.forward(i.reshape(1,-1))
#     print(model.layers[-1].A)
#print(model.forward(x))

[accuracy,f1_score,confusion_matrix] = model.evaluate(X_test,label_test,metric = ['accuracy','f1 score','confusion matrix'])
print(f"accuracy: {accuracy}")
print(f"f1_score: {f1_score}")
print(f"confusion_matrix: {confusion_matrix}")