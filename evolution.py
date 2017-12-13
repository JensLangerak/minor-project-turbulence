#evolution process

from cgp import *
import math
import random
import numpy as np

#fitness score calculation
def calcFitness(output, ref):
    a = output[0] - ref[0]
    b = output[1] - ref[1]
    return math.sqrt(a * a + b * b)

half_population = 20
population = 2 * half_population
nr_nodes=5
node_size=3
mutation_chance = 0.05
max_error = 0.1

#stage 1: create solutionLists
solutionLists=population*[None]
for i in range(population):
    solutionLists[i]=random.sample(range(0,50),nr_nodes*node_size)
    


#stage 2: calculate outputs
ref =       [[0, 0],       [1,1],          [4,4],     [9,9]]
features = [[0,1,2],      [1,2,3],      [2,1,3],     [3,0,0]]

Fitnesslist=population*[None]
smallestError = 100
stop = 0;
for g in range(100):
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
        
    nextGeneration = population*[None]
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
    
    
    solutionLists = nextGeneration
    print("Generation ", g, " error ", Fitnesslist[0][1], " avg ", sum(Fitnesslist[:][1]) / len(Fitnesslist[:][1]))
    

print (cgp(features[0], result))
print (cgp(features[1], result))
print (cgp(features[2], result))
print (cgp(features[3], result))
#random selection of previous solutions based on fitness score
