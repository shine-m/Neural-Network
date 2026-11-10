import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class dense_layer:
    def __init__(self, n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)*.01
        # to create n_inputs x n_neurons weight matrix
        self.bias = np.zeros((1,n_neurons))

    def forward (self ,inputs):
        self.output = np.dot(inputs, self.weights)+ self.bias



nnfs.init()
x,y = spiral_data(samples=100, classes =3)

dense1 = dense_layer(2,3)
dense1.forward(x)

print(dense1.output[:10])