import nn

model = nn.load("model.NND")
print(model.layers[0].weights)