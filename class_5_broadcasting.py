import numpy as np


a= [[1,2,3],[4,5,6],[7,8,9]]


a = np.array(a)

print(np.max(a, axis = 0,keepdims=1))

# we will subtract the max of each row from all the elements of that row 

print(a - np.max (a, axis=1, keepdims= True))