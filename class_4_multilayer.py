import numpy as np

weights1 = [[1,2,3,0],[1,3,2,2],[.5,2,.10,1]]
weights2 = [[.5,.2,1],[2,3,.2],[1.5,2.5,.50]]

inputs= [[2,3,4,5], [3,2,5,2], [2,5,6,1],[1,2,3,6], [1,0,2,1]]

bias1 = [2,1,0]
bias2 = [ .5,2,1]

weights1 = np.array(weights1)
weights2 = np.array(weights2)
inputs= np.array(inputs)

output1 = np.dot(inputs, weights1.T) + bias1

print(output1)

output2 = np.dot(output1, weights2.T)+bias2
print(output2)