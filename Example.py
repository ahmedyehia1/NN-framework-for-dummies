import numpy as np
import nn

model = nn.Model(
     nn.Layer(size=(2,1), activation='Relu')
     #nn.Layer(size=(2,3), activation='Relu'),
     #nn.Layer(size=(3,1), activation='sigmoid')
     #nn.Layer(size=(10,6), activation='ReLU'),
     #nn.Layer(size=(6,1), activation='ReLU')

)

x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])

label = np.array([[0], [1], [1], [1]])

'''
x = np.array([
    [2,5,2],
    [5,3,1],
    [4,2,6]
]
)'''
test_x = np.array([
    [2,-5,2],
    [5,4,-1]
])
test_y = np.array([[0],[1]])




#print(model.forward(x))
model.fit(x,label,'SGD','MSE',alpha = 0.0001,epoch = 10000,graph_on = True)
print(model.layers[0].Z)
for i in x:
    model.forward(i.reshape(1,-1))
    print(model.layers[-1].A)
#print(model.forward(x))

# accuracy = model.evaluate(test_x,test_y)
# print(accuracy)