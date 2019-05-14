from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, relu
from scipy.cluster.hierarchy import dendrogram, linkage

class ItemRelationSeparatedNet(nn.Module):
    def __init__(self, rep_size, hidden_size, ncuisines, ntastes, output_shape):
        super(ItemRelationSeparatedNet, self).__init__() 
        
        # Taste to representation 
        self.t2r = nn.Linear(ntastes, rep_size)
        
        # Representation to hidden
        self.rep2h = nn.Linear(rep_size, hidden_size)
        
        # Relation to hidden
        self.rel2h = nn.Linear(ncuisines, hidden_size)
        
        # Hidden to attributes (ingredients)
        self.h2o = nn.Linear(hidden_size, output_shape)
        

    def forward(self, x):
        
        x = x.view(-1,self.ncuisines+self.ntastes) # reshape as size [B x (nobj+nrel) Tensor] if B=1
        x_cuisine = x[:,:self.ncuisines]
        x_taste = x[:,self.ncuisines:self.ncuisines+self.ntastes] # input to Item Layer [B x nobj Tensor]
        
        repLayer_input_taste = relu(self.t2r(x_taste))
        hidLayer_input_cuisine = x_cuisine
        
        repTaste = relu(repLayer_input_taste)
        
        hidden = relu(self.rep2h(repTaste) + self.rel2h(hidLayer_input_cuisine))
        output = self.h2o(hidden)
        output = sigmoid(output)

        return output, hidden, repTaste
    
    
class BaselineFeedForward(nn.Module):
    def __init__(self, rep_size, hidden_size, ncuisines, ntastes, output_shape):
        super(BaselineFeedForward, self).__init__()
         
        self.ncuisines = ncuisines
        self.ntastes = ntastes
        self.output_shape = output_shape
        self.i2r = nn.Linear(ncuisines+ntastes, rep_size)
        self.r2h = nn.Linear(rep_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        
        x = x.view(-1,self.ncuisines+self.ntastes) 
        x_item = x[:,:self.ncuisines+self.ntastes] 
        
        rep = self.i2r(x_item)
        rep = relu(rep)
       
        hidden = self.r2h(rep)
        hidden = relu(hidden)
        output = self.h2o(hidden)
        output = sigmoid(output)

        return output, hidden, rep
    
class ExtensionFeedForward(nn.Module):
    def __init__(self, rep_size, hidden_size, ncuisines, ntastes, output_shape):
        super(ExtensionFeedForward, self).__init__()
          
        self.ncuisines = ncuisines
        self.ntastes = ntastes
        self.output_shape = output_shape

        self.i2r = nn.Linear(ncuisines+ntastes, rep_size)
        self.r2h1 = nn.Linear(rep_size, hidden_size)
        self.h1_2_h2 = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        
        x = x.view(-1,self.ncuisines+self.ntastes) 
        x_item = x[:,:self.ncuisines+self.ntastes] 
        
        rep = self.i2r(x_item)
        rep = relu(rep)
        
        hidden = self.r2h1(rep)
        hidden = relu(hidden)

        hidden2 = relu(self.h1_2_h2(hidden))
        output = self.h2o(hidden2)
        output = sigmoid(output)
   
        return output, hidden2, rep