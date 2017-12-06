# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:46:08 2017

@author: jens
"""

node_size = 3
nr_nodes = 10
outputs = 2
operations = 4

def cgp(features, solutionList):
    inputs = [0] * (len(features) + nr_nodes)
    for i in range(len(features)):
        inputs[i] = features[i]
    
    maxInput = len(features)
    
    for i in range(nr_nodes):
        a_index = solutionList[i * node_size]
        b_index = solutionList[i * node_size + 1]
        o_index = solutionList[i * node_size + 2]
    
        a = inputs[a_index % maxInput]
        a = inputs[b_index % maxInput]
    