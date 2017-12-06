#evolution process

from cgp.py import *
import random

population = 10
nr_nodes=5
node_size=3


#stage 1: create solutionLists
solutionLists=population*[None]
for i in range(population):
    solutionLists[i]=random.sample(xrange(1,50),nr_nodes*node_size)
    


#stage 2: calculate outputs
ref = [0, 0]

Fitnesslist=population*[None]
for i in range(population):
    output = cgp(features, solutionLists[i]) #x-coordinate
    Fitnesslist = (i, calcFitness(output, ref))
    Fitnesslist.sort(key=lambda x: x[1])



#fitness score calculation
def calcFitness(output, ref)
    return np.linalg.norm(output - ref)

#random selection of previous solutions based on fitness score
