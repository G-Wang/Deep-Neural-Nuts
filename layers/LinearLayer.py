
# coding: utf-8

# In[1]:

import numpy as np
from utils import Make_W, Sigmoid, DSigmoid

class LinearLayer:
    """ A fully connected, feed forward linear layer utilzing eith Sigmoid, Tanh or Relu Transfer function
        Input and output to the data is assumed to be [1 x input_dim] and [1 x output_dim] respectively
    """
    @staticmethod
    def init(input_dim, hidden_dim, output_dim, batch_size, transfer_func):
        model = {}
        # create a dictionary that will store the activation, Weights, bias, and their respective gradients 
        model['In'] = np.zeros((batch_size, input_dim)) # input feeding to the layer
        model['Grad_In'] = np.zeros_like(model['In']) # gradient of the input weights
        model['Hidden'] = np.zeros((batch_size, hidden_dim)) # hidden activation
        model['Grad_Hidden'] = np.zeros_like(model['Hidden']) # gradient of hidden activation
        model['Out'] = np.zeros((batch_size,output_dim)) # output activations from the layer
        model['Grad_Out'] = np.zeros_like(model['Out']) # gradient of output activation
        model['Win'] = Make_W(input_dim, hidden_dim) # weight matrix from input to hidden
        model['bin'] = np.zeros((batch_size, hidden_dim)) # bias matrix from input the hidden
        model['D_Win'] = np.zeros_like(model['Win']) # derivative weight matrix from input to hidden
        model['D_bin'] = np.zeros_like(model['bin']) # derivative of bias matrix from input to hidden
        model['Wout'] = Make_W(hidden_dim, output_dim) # weight matrx from hidden to output
        model['bout'] = np.zeros((batch_size, output_dim)) # bias matrix from hidden to output
        model['D_Wout'] = np.zeros_like(model['Wout']) # derivative of weight matrix from hidden to output
        model['D_bout'] = np.zeros_like(model['bout']) # derivative of weight matrix from hidden to output
        if transfer_func == 'Sigmoid':
            model['TF'] = Sigmoid
            model['D_TF'] =DSigmoid
        #elif transfer_fun = 'Tanh':
        #    model['TF'] = 'Tanh
        return model
    
    @staticmethod
    def forward(model, Input):
        # assumes the input is of dimension [batch_size x input_dim], forward the layer for one batch of examples
        model['In'] = Input
        model['Hidden'] = model['In'].dot(model['Win']) + model['bin'] # Hidden = Input*Win + Bin
        model['Hidden'] = model['TF'](model['Hidden']) # apply the nonlinear transfer function activation
        model['Out'] = model['Hidden'].dot(model['Wout']) + model['bout'] # Out = Hidden*Wout + Bout
        model['Out'] = model['TF'](model['Out']) # apply the nonlinear transfer function activation
        return model

