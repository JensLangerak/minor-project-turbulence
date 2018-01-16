#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:39:09 2017

@author: mikael
"""

# read DNS

import sys
sys.path.append("..")
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import PyFOAM as pyfoam
import scipy.interpolate as interp
from tempfile import mkstemp
from shutil import move
from shutil import copy
from os import remove, close
import os

def loadData_avg(dataset):

    with open(dataset, 'rb') as f:
        reader = csv.reader(f)
        names = reader.next()
        ubulk = reader.next()
        data_list = np.zeros([len(names), 100000])
        for i,row in enumerate(reader):
            if row:
                data_list[:,i] = np.array([float(ii) for ii in row])
    data_list = data_list[:,:i+1] 
    data = {}
    for j,var in enumerate(names):
        data[var] = data_list[j,:]
              
    return data

def interpDNSOnRANS(dataDNS, mesh):
    
    names = dataDNS.keys()    
    data = {}
    xy = np.array([dataDNS['Z'], dataDNS['Y']]).T
#    print(xy)
#    print(dataDNS['um'])
    for var in names:
        if not var=='Z' and not var=='Y':        
            data[var] = interp.griddata(xy, dataDNS[var], (mesh[0], mesh[1]), method='linear')
#    print('shape xy: ' + str(xy.shape))
    return data


data = loadData_avg('/home/mikael/OpenFOAM/mikael-3.0.1/run/SquareDuct/DATA/03500_full.csv')

data['k'] = 0.5 * (data['uu'] + data['vv'] + data['ww'])

Ny = 100
Nz = 100

y = np.linspace(-1,1,Ny)
z = np.linspace(-1,1,Nz)
Y, Z= np.meshgrid(y, z, sparse=False, indexing='ij')
mesh = [Y,Z]

data_mesh = interpDNSOnRANS(data, mesh)

data_mesh['tau'] = np.zeros([3,3,Ny,Nz])
data_mesh['bij'] = np.zeros([3,3,Ny,Nz])

data_mesh['tau'][0,0,:,:] = data_mesh['uu']
data_mesh['tau'][1,1,:,:] = data_mesh['vv']
data_mesh['tau'][2,2,:,:] = data_mesh['ww']
data_mesh['tau'][0,1,:,:] = data_mesh['uv']
data_mesh['tau'][1,0,:,:] = data_mesh['uv']
data_mesh['tau'][0,2,:,:] = data_mesh['uw']
data_mesh['tau'][2,0,:,:] = data_mesh['uw']         
data_mesh['tau'][1,2,:,:] = data_mesh['vw']
data_mesh['tau'][2,1,:,:] = data_mesh['vw']

for i in range(Ny):
    for j in range(Nz):
        data_mesh['bij'][:,:,i,j] = data_mesh['tau'][:,:,i,j]/(2.*data_mesh['k'][i,j]) - np.diag([1/3.,1/3.,1/3.])
        
cmap=plt.cm.coolwarm
cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])

plt.figure()
contPlot = plt.contourf(mesh[0], mesh[1], data_mesh['um'],50,cmap=cmap,extend="both")
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.title('DNS:  $\overline{U}$')
plt.colorbar(contPlot)
plt.show()

plt.figure()
contPlot = plt.contourf(mesh[0], mesh[1], data_mesh['bij'][0,0,:,:],50,cmap=cmap,extend="both")
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.title('$b_{11}$')
plt.colorbar(contPlot)
plt.show()

plt.figure()
contPlot = plt.contourf(mesh[0], mesh[1], data_mesh['bij'][1,1,:,:],50,cmap=cmap,extend="both")
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.title('$b_{22}$')
plt.colorbar(contPlot)
plt.show()

plt.figure()
contPlot = plt.contourf(mesh[0], mesh[1], data_mesh['bij'][2,2,:,:],50,cmap=cmap,extend="both")
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.title('$b_{33}$')
plt.colorbar(contPlot)
plt.show()

plt.figure()
contPlot = plt.contourf(mesh[0], mesh[1], data_mesh['bij'][0,1,:,:],50,cmap=cmap,extend="both")
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.title('$b_{12}$')
plt.colorbar(contPlot)
plt.show()

plt.figure()
contPlot = plt.contourf(mesh[0], mesh[1], data_mesh['bij'][0,2,:,:],50,cmap=cmap,extend="both")
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.title('$b_{13}$')
plt.colorbar(contPlot)
plt.show()

plt.figure()
contPlot = plt.contourf(mesh[0], mesh[1], data_mesh['bij'][1,2,:,:],50,cmap=cmap,extend="both")
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.title('$b_{23}$')
plt.colorbar(contPlot)
plt.show()