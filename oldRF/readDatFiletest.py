
import numpy as np
#import scipy as sp
#import itertools
#from scipy.sparse import csr_matrix
import csv as csv
#import matplotlib
#matplotlib.use('Qt5Agg') 

import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import scipy.interpolate as interp
from scipy import spatial
#import scipy.stats as st
#from scipy.optimize import curve_fit
#import mplrc.ieee.transaction
#import os, commands
from tempfile import mkstemp
from shutil import move
from shutil import copy
from os import remove, close
import subprocess
import collections
import openFOAMCD as foam
#import pickle

filename = '../DATA/conv-div-mean-half.dat'

def loadData_avg(dataset):

    #amount of lines between different blocks
    ny = 193
    nx = 1152
    
    interval = int(np.ceil((float(ny)*float(nx))/5.0))
    
    dataFile = {}
    dataFile['vars'] = ["x","y","U","V","W","dx_mean_u","dx_mean_v","dx_mean_w","dy_mean_u","dy_mean_v",
    "dy_mean_w","dz_mean_u","dz_mean_v","dz_mean_w","uu","uv","uw","vv","vw","ww"]
    dataFile['vals'] = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    l_start = np.zeros(len(dataFile['vars']),dtype=int)
    l_end = np.zeros(len(dataFile['vars']),dtype=int)
    start_data = 26
    
    # datafile is separated in blocks, each belonging to one of the variables.
    # set-up the start and end values of each block
    for i in range(len(dataFile['vars'])):
        l_start[i] = start_data + i*interval
        l_end[i] = (interval+start_data-1) + i*interval
    
    with open(filename, 'rb') as f:
        
        # go through all lines of file
        for i,line in enumerate(f):  
            
            # go through all variables
            for j in range(l_start.shape[0]):
                
                # check the variable the current block belongs to
                if i >= l_start[j] and i <= l_end[j]:
                    dataFile['vals'][j].append([float(x) for x in line.split()])
    
    data_DNS = {}
    data = {}
    # flatten the obtained lists with data            
    for i in range(len(dataFile['vals'])):
        data_DNS[dataFile['vars'][i]] = np.array([item for sublist in dataFile['vals'][i] for item in sublist])
        data[dataFile['vars'][i]] = np.reshape(data_DNS[dataFile['vars'][i]],[nx,ny])
    
    return data

case = 'ConvergingDivergingChannel'
dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
            
dataDNS = foam.loadData_avg(case, dataset)
#dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
#dataDNS_i['k'] = 0.5 * (dataDNS_i['uu'] + dataDNS_i['vv'] + dataDNS_i['ww'])


print('np.shape(data) = ', np.shape(dataDNS))
print(np.shape(dataDNS['y']))
# verify plots
cmap=plt.cm.coolwarm
cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])

plt.figure()
contPlot = plt.contourf(dataDNS['x'], dataDNS['y'], dataDNS['uu'],50,cmap=cmap,extend="both")
#plt.axis('equal')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('bla')
plt.colorbar(contPlot)
plt.show()




