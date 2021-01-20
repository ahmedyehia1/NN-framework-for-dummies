import numpy as np

def save_weights(layer_arr):
    f = open('saved_weights.csv', 'w')
    for layer in layer_arr:
        f.write(",".join(str(x) for x in layer) + "\n")

# save_weights(np.array([[1,2,3,4,56,9,7],[3,4,6,7,8]]))

def import_saved_weights():
    all_of_the_weights = open('saved_weights.csv', 'r').read()
    lines = all_of_the_weights.split('\n')
    layer_arr = []
    for line in lines:
        if len(line) > 1:
            x = line.split(',')
            xx = np.array(x)
            y = xx.astype(np.float)
            layer_arr.append(y)
    return layer_arr

# print(import_saved_weights())
