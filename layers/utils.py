
# coding: utf-8

# In[ ]:

from random import uniform
import numpy as np

def Make_W(in_dim, out_dim):
    Num = 0.1
    return (np.random.rand(in_dim, out_dim) * 2 -1)*Num # Creates a matrix array of size [input_dim * output_dim]

def Sigmoid(x):
    return 1./(1+np.exp(-x))

def DSigmoid(x):
    return Sigmoid(x)*(1-Sigmoid(x))


