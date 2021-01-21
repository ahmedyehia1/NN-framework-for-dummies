import numpy as np
import nn

model = nn.Model(
    nn.Layer(size=(3,5), activation='ReLU'),
    nn.Layer(size=(5,10), activation='ReLU'),
    nn.Layer(size=(10,6), activation='ReLU'),
    nn.Layer(size=(6,1), activation='ReLU')
)

x = np.array([
    [2,5,2],
    [5,3,1],
    [4,2,6]
]
)
test_x = np.array([
    [2,-5,2],
    [5,4,-1]
])
test_y = np.array([[0],[1]])
label = np.array([[50] , [70] , [40]])

#y_hat = model.forward(x)


# not completed yet
model.fit(x,label,'SGD','MSE',1,0.01)
##

# accuracy = model.evaluate(test_x,test_y)
# print(accuracy)