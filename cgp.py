# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:46:08 2017

This file is able to execute a Cartesian Genetic Programming program (cgp). The program can be represented by a list of
integers. This list representation of integers is called the solutionList or chromosome. In this list a node is
represented by 3 integers. The first two integers refers to the input indices and the third integer translates to the
operation that should be performed with the inputs.

A node refers to a single operation in a cgp program. This consist of two inputs and an operation.
The result of <input 1> <operation> <input 2> is the output of the node and can be used in any successive node.
Features are the input values for the entire model, this are the only inputs that the first node can use.
Inputs refer to any value that a node (single operation) can use for its computation. The possible inputs are all the
features and the result of the previous nodes. The final input list contains all the features and all the outputs of the
nodes.

Input_id refers to the location in the input list.
Node_id refers to the location in the chromosome or translated list. (note since a node contains of node_size elements
this id should be multiplied by node_size)

The output are the outputs of the last #outputs (currently 2) nodes.
"""

import math
import numpy as np
node_size = 3  # How many numbers are needed for one node. Two for the input and one to determine the operation.
outputs = 2  # Number of outputs that should be returned.
operations = 4  # Number of supported operations.

operation_list = ['+', '-', '*', '/']

def translate(nr_features, chromosome):
    """
    Translate a list of numbers (chromosome) into an representation of a cartesian genetic program.
    :param nr_features: number of feature that will be used in the genetic program.
    :param chromosome: The solutionList, List of numbers that should be translated into a cartesian genetic program.
    :return: A list representation of a cartesian genetic program.
    """
    nr_nodes = math.floor(len(chromosome) / node_size)

    translated_list= [0] * len(chromosome) # Create new list in advance (is faster than with append)
    for i in range(nr_nodes):  # Turns numbers into the right numbers/operation
        # starts with 0 and can be used to determine how many outputs of nodes can be used as input.
        translated_list[node_size*i  ] = chromosome[node_size*i  ] % (nr_features+i)  # input 1
        translated_list[node_size*i+1] = chromosome[node_size*i+1] % (nr_features+i)  # input 2

        translated_list[node_size*i+2] = operation_list[chromosome[node_size*i+2] % operations]  # operaion
    return translated_list


def translate_item(nr_features, item_index, item_value):
    node_index = math.floor(item_index / node_size)
    sub_index = item_index - node_size * node_index
    if sub_index == 2:
        return operation_list[item_value % operations]
    else:
        return item_value % (nr_features + node_index)


def cgp(features, chromosome):  #function that interpretes solution in terms of features -> returns two outputs
    """
    Functions that translates a chromosome to a cgp program and applies the features to that program. Returns the output
    of the last #outputs nodes.
    :param features: Features that should be used to calculate the result.
    :param chromosome: The solutionList, List of numbers that should be translated to a cartesian genetic program.
    :return: The output of the last #outputs nodes.
    """
    nr_nodes = math.floor(len(chromosome) / node_size)  # number of nodes in the chromosome
    nr_features = len(features)
    nr_inputs = nr_nodes + nr_features

    if nr_inputs < outputs:
        raise ValueError("Number of possible outputs is smaller than the number of desired outputs")

    # Inputs is used to keep track of the input values that are already calculated (or known at the start). This list
    # makes sure the each node is evaluated at most once. When it start it only knows the values of the features.
    # During the computation the list is filled with the output of the nodes.
    inputs = [None] * nr_inputs  # Create list. None means that the value is not yet known.
    for i in range(nr_features):
        inputs[i] = features[i]  # Copy the features into the input list.

    #translated_list = translate(nr_features, chromosome) # translate the chomoso\mne into a cgp program.
    translated_list = chromosome
    # get outputs number of outputs. The first output is the last node, the second output is the second-last node etc.
    result = []
    for i in range(1, outputs + 1):
        # Get the output of the last + 1 - i node and append it to the result list. (nr_inputs is last + 1)
        # The inputs list contains the values of all computed nodes and is used to pass these values to successive calls
        # of calculate_input
        result.append(calculate_input(translated_list, nr_inputs - i, inputs, nr_features))

    return result


def calculate_input(cgp_program, input_id, inputs, nr_features):
    """
    Get the value for input input_id. At the end of the function, the outputs of the nodes that are evaluated are stored
    in inputs.
    :param cgp_program: A cgp program, stored as list. Is the translation of a chromosome
    :param input_id: ID of the input that must be returned. The first ids are the features [0, #features - 1] and the
    other ids are the outputs of the nodes [#features, #features + #nodes - 1]
    :param inputs: List that keeps track of all computed inputs. At the start it only contains the values for the
    inputs. Currently unknown values have the value None.
    :param nr_features: #features in inputs
    :return: the values of inputs[input_id]
    """

    if inputs[input_id] is not None:  # value is already known, return value
        return inputs[input_id]

    node_id = input_id - nr_features  # inputs[input_id] is not yet known, thus is not a feature, calculate the node_id

    # Get the input index for the two inputs of this node.
    a_index = cgp_program[node_size*node_id]
    b_index = cgp_program[node_size*node_id+1]

    # Get the value of the two inputs
    a = calculate_input(cgp_program, a_index, inputs, nr_features)
    b = calculate_input(cgp_program, b_index, inputs, nr_features)

    # Get the operation.
    o = cgp_program[node_size * node_id + 2]

    # Calculate the output of this node.
    if o =='+':
        output = a+b
    elif o =='-':
        output = a-b
    elif o =='*':
        output = a*b
    else:
        if (b > 0) :
            off = 0.00001
        else:
            off = -0.00001
        output=a/(b + off) #safe division

    # Store the output in the inputs list.
    inputs[input_id] = output
    return output

#nr_features = len(features)
#lst=translate(nr_features, chromosome)
#length=len(lst)
#nrnodes=length/node_size
#nodelist=nrnodes*[0]
#output1ref=nrnodes-2
#output2ref=nrnodes-1

    
def nodes_used(outputref, nodelist,lst):
    if outputref< nr_features :
        return nodelist
    outputref-=nr_features
    #print(outputref)
    if nodelist[outputref]==1:
        return nodelist
    nodelist[outputref]=1
    #print(lst)
    inputs=[lst[node_size*(outputref)], lst[node_size*(outputref)+1]]
    #print("inputs")
    #print(inputs)
    nodes_used(inputs[0],nodelist, lst)
    nodes_used(inputs[1],nodelist, lst)
    return nodelist
    
#test
#sol=[1,1,'*',1,1,'+', 2,2,'+', 1, 2, '*',1,4,'*',2,4,'/', 5,2,'+', 1, 1, '*'] #should result in [1,0,0,1,0,0,1,1]
#sol=[1, 3, '-', 2, 2, '+', 1, 0, '+', 3, 1, '*', 5, 0, '-', 1, 4, '+', 5, 7, '-', 7, 4, '+', 7, 7, '-', 7, 12, '+', 13, 2, '-', 6, 12, '-', 3, 15, '-', 12, 13, '*', 17, 4, '-', 14, 12, '-', 19, 8, '*', 11, 5, '-', 12, 19, '-', 10, 5, '+']


#nr_features=4

def createListnodes(sol, nr_features):
    #print("sol")
    #print(sol)
    length=len(sol)
    nrnodes=int(length/node_size)
    nodelist=nrnodes*[0]
    output1ref=nrnodes-2
    output2ref=nrnodes-1
    onlynodes=nodes_used(output1ref+nr_features, nodelist, sol)
    solution=nodes_used(output2ref+nr_features, onlynodes, sol)
    #print("Tes")
    #print (solution)
    #turn nodelist into list size of solution:
    newlist=nrnodes*node_size*[0]
    for i in range(nrnodes):
        if solution[i] is None:
            print ('testing')
            print (sol)
        if solution[i]==1:
            for j in range(node_size):
                newlist[node_size*i]=1
                newlist[node_size*i+j]=1
                newlist[node_size*i+j]=1  
    return newlist

def complete_translate(cgp_program, nr_features, nr_nodes):
    completeTranslate = (nr_features + nr_nodes) * [""]
    for i in range(nr_features):
        completeTranslate[i] = "f_g[offset+" + str(i) + "]"

    for n in range(nr_nodes):
        base = n * node_size
        d = n + nr_features
        completeTranslate[d] = "(" + completeTranslate[cgp_program[base]] + " " + cgp_program[base + 2] + " " + \
                               completeTranslate[cgp_program[base + 1]] + ")"

    return completeTranslate

def complete_translate_to_ints(cgp_program):
    res = np.empty_like(cgp_program)
    for i in range(len(cgp_program)):
        k = cgp_program[i]
        if k == '+':
            k = 0
        if k == '-':
            k = 1
        if k == '*':
            k = 2
        if k == '/':
            k = 3
        res[i] = k
    return res




#createListnodes(sol, nr_features)

            
        

#main program

#newlist=[1,2,'+',4,0,'*']
#features=[1,2,3,4]
#input_id=5
#print( calculateInput(newlist, input_id, features))

    
#cgp(features, [1,2,0, 1,2,0, 4,3,2])


