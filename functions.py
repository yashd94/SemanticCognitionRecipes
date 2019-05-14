from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, relu
from scipy.cluster.hierarchy import dendrogram, linkage

def train(mynet, epoch_count, N, input_pats, output_pats, optimizer, criterion = nn.MSELoss(), nepochs_additional = 5000, verbose = True):
    # Input
    #  mynet : Net class object
    #  epoch_count : (scalar) how many epochs have been completed so far
    #  nepochs_additional : (scalar) how many more epochs we want to run
    mynet.train()
    for e in range(nepochs_additional): # for each epoch
        error_epoch = 0.
        perm = np.random.permutation(N)
        for p in perm: # iterate through input patterns in random order
            mynet.zero_grad() # reset gradient
            output, hidden, rep = mynet(input_pats[p,:]) # forward pass
            target = output_pats[p,:] 
    
            loss = criterion(output, target) # compute loss
            
            loss.backward() # compute gradient 
            optimizer.step() # update network parameters
            error_epoch += loss.item()
        error_epoch = error_epoch / float(N)
        if verbose:
            if e % 500 == 0:
                print('epoch ' + str(epoch_count+e) + ' loss ' + str(round(error_epoch,3)))
    return epoch_count + nepochs_additional

def get_rep(net, item_names):
    # Extract the hidden activations on the Representation Layer for each item
    # 
    # Input
    #  net : Net class object
    #
    # Output
    #  rep : [nitem x rep_size numpy array], where each row is an item
    input_clean = torch.zeros(26, 26)
    for idx,name in enumerate(item_names):
        input_clean[idx,idx] = 1. # 1-hot encoding of each object (while Relation Layer doesn't matter)
    output, hidden, rep = net(input_clean)
    return rep.detach().numpy()

def plot_rep(rep1, rep2, rep3, names, label_rotation = 0):
    #  Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    #  using bar graphs
    # 
    #  Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    #  names : [nitem list] of item names
    #
    nepochs_list = [nepochs_phase1, nepochs_phase2, nepochs_phase3]

    nrows = rep1.shape[0]
    R = np.dstack((rep1, rep2, rep3))    
    mx = R.max()
    mn = R.min()
    depth = R.shape[2]
    count = 1
    plt.figure(1,figsize=(30, 20))
    for i in range(nrows):
        for d in range(R.shape[2]):
            plt.subplot(nrows, depth, count)
            rep = R[i,:,d]
            plt.bar(range(rep.size),rep)
            plt.ylim([mn,mx])
            plt.xticks([])
            plt.yticks([])        
            if d==0:
                plt.ylabel(names[i], rotation = label_rotation, size = 15)
            if i==0:
                plt.title(str(nepochs_list[d]) + " epochs", size = 15)
            count += 1
    plt.suptitle("Cuisine representations over epochs", fontsize = 20)
    plt.show()

def plot_dendo(rep1, rep2, rep3, names, epoch_title, x_rotation = 0, y_rotation = 0):
    #  Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    #  using hierarchical clustering
    # 
    #  Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    #  names : [nitem list] of item names
    #
#     nepochs_list = [nepochs_phase1, nepochs_phase2, nepochs_phase3]
    linked1 = linkage(rep1,'single')
    linked2 = linkage(rep2,'single')
    linked3 = linkage(rep3,'single')
    mx = np.dstack((linked1[:,2], linked2[:,2], linked3[:,2])).max()+0.1    
    plt.figure(2, figsize=(5, 10))
#     plt.subplot(1,3,1)    
#     plt.suptitle('Differention in learning over epochs', fontsize = 20)
#     dendrogram(linked1, labels = names, color_threshold = 0, leaf_rotation = 90, orientation = 'left')
#     plt.ylim([0,mx])
#     plt.title(str(nepochs_list[0])+" epochs", size = 15)
#     plt.ylabel('Euclidean distance', size = 12)
#     plt.subplot(1,3,2)
#     plt.title(str(nepochs_list[1])+" epochs", size = 15)
#     dendrogram(linked2, labels=names, color_threshold=0, leaf_rotation = 90, orientation = 'left')
#     plt.ylim([0,mx])
#     plt.subplot(1,3,3)
    plt.title(str(epoch_title)+" epochs", size = 15)
    dendrogram(linked3, labels = names, color_threshold = 0, leaf_rotation = 90, orientation = 'left')
    plt.xticks(rotation = x_rotation)
    plt.yticks(rotation = y_rotation)
    plt.show()