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

# import and preprocess data
x,label = Data.get_data("data_banknote_authentication.csv")
x = Data.normalize(x)
X_train, X_test, label_train, label_test = Data.split_data(x,label)

# Train the model
model.fit(X_train,label_train,'SGD','MSE',alpha = 0.0001,epoch = 15,graph_on = True)

# evaluate the model
[accuracy,f1_score,confusion_matrix] = model.evaluate(X_test,label_test,metric = ['accuracy','f1 score','confusion matrix'])
print(f"accuracy: {accuracy}")
print(f"f1_score: {f1_score}")
print("confusion matrix:\n",confusion_matrix)

model.save()