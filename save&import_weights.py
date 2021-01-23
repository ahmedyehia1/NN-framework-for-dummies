import numpy as np

def save_weights(layer_arr):
    
    '''
        saves 2d array into csv file:
        Args:
            layer_arr: a 2d array holds numbers
        Shape:
            - Input: layer_arr (N, m) N: number of weights
            - Output: None
        Examples:
            save_weights(np.array([[1,2,3,4,56,9,7],[3,4,6,7,8]]))
        '''
    
    f = open('saved_weights.csv', 'w')
    for layer in layer_arr:
        f.write(",".join(str(x) for x in layer) + "\n")

def import_saved_weights():
    
    '''
        import 2d array from csv file:
        Args: None
        Shape:
            - Input: None
            - Output: layer_arr (N, m) N: number of weights
        Examples:
            import_saved_weights()
    '''    
    
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
