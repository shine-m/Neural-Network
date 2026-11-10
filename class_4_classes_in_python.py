

import numpy as np

class animal :
    def __init__(self, family, genes):
        self.family = family
        self.genes = genes
        self.weights =  np.random.randn(3,4)*.01


newobj = animal ("dog","animalia")

print(newobj.genes)
print(newobj.family)

newobj.ans = 4545 #  class properties can be added even outside of the class

print(newobj.ans)
print(newobj.weights)