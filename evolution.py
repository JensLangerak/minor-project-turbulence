#evolution process

from cgp import *
import math
import random
import numpy as np

#fitness score calculation
def calcFitness(output, ref):
    a = output[0] - ref[0]
    b = a#output[1] - ref[1]
    return math.sqrt(a * a + b * b)

def diversity(solutionList):
    dim = np.shape(solutionList)
    k = 0
    for j in range(dim[1]):
        symbols = 60 * [0]
        for i in range(dim[0]):
            if (symbols[solutionList[i][j]] == 0):
                k += 1
                symbols[solutionList[i][j]] = 1
    
    return k / dim[1] / (2 * half_population)

def average(Fitnesslist):
    sumFit = 0
    for i in range(population):
        sumFit += Fitnesslist[i][1]
    return sumFit / population

half_population = 30
population = 2 * half_population + 1 
nr_nodes=20
node_size=3
mutation_chance = 0.10
max_error = 0.01

#stage 1: create solutionLists
solutionLists=population*[None]
for i in range(population):
    solutionLists[i]=np.random.randint(0,50, nr_nodes*node_size)
    


#stage 2: calculate outputs
ref =  []
features = []
for i in range(100):
    features += [[i]]
    res = 3 * i ** 3 - 2 * i ** 2 +  i
    ref += [[res, res]]

Fitnesslist=population*[None]
smallestError = 10000000
result = solutionLists[0]
stop = 0;
for g in range(500):
    if (smallestError < max_error):
        break
    
    for i in range(population):
        fitness = 0.0
    #    print (solutionLists[i])
        for j in range(len(features)) : 
            output = cgp(features[j], solutionLists[i])
            fitness += calcFitness(output, ref[j])
        Fitnesslist[i] = (i, fitness / len(features))
        

    Fitnesslist.sort(key=lambda x: x[1])
    
    if (Fitnesslist[0][1] < smallestError):
            result = solutionLists[Fitnesslist[0][0]]
            smallestError = Fitnesslist[0][1]
            print (translate(features[0], solutionLists[Fitnesslist[0][0]]))
            print (smallestError)
            if (smallestError < max_error):
                print("klaar")
                result = solutionLists[Fitnesslist[0][0]]
                break
                
                
    selectionList = []
    for i in range(population):
        selectionList += (population - i) * [Fitnesslist[i][0]]
        
    nextGeneration = (population)*[None]
    for i in range(half_population):
        p1 = selectionList[random.randint(0,len(selectionList) - 1)]
        p2 = selectionList[random.randint(0,len(selectionList) - 1)]
        cross = random.randint(1, nr_nodes*node_size)
        ps1 = solutionLists[p1]
        ps2 = solutionLists[p2]
        c1= nr_nodes*node_size * [None]
        c2 = nr_nodes*node_size * [None]
        for j in range (nr_nodes*node_size):
            m1 = random.uniform(0, 1)
            m2 = random.uniform(0, 1)
            if (j < cross):
                c1[j] = ps1[j]
                c2[j] = ps2[j]
            else:
                c1[j] = ps2[j]
                c2[j] = ps1[j]
            if (m1 < mutation_chance):
                c1[j] = random.randint(0, 50)
                
            if (m2 < mutation_chance):
                c2[j] = random.randint(0, 50)
        nextGeneration[2 * i] = c1
        nextGeneration[2 * i + 1] = c2
    
    nextGeneration[population - 1] = result

    
    print("Generation ", g, " error ", Fitnesslist[0][1], " med ", Fitnesslist[half_population][1], " div ", diversity(solutionLists))

    solutionLists = nextGeneration

print (cgp(features[0], result))
print (cgp(features[1], result))
print (cgp(features[2], result))
print (cgp(features[3], result))
#random selection of previous solutions based on fitness score
