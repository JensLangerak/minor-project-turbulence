# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:09:45 2017

@author: thomas
"""

from __future__ import division
import numpy as np
import os
import math
import matplotlib.pyplot as plt


time_end      = 30000 

#Specify the reynoulds number and turbulence model used (directory)
Re            = 700 #
TurbModel     = 'kOmega'

# What are these???????????
nx_RANS       = 140
ny_RANS       = 150

#Specify home directory from where the data can be found
dir = os.path.dirname(__file__)
home = os.path.realpath('MinorCSE') + '/'
#home = '../CSE minor/'

dir_RANS  = home + ('Re%i_%s' % (Re,TurbModel))




##################################################################################



#This function reads the data from a scalar RANS file (for example the pressure p) 
#and returns an array v with 1 columns and "maxLines" rows
def getRANSScalar(case, time, var):
    #file = open(case + '/0/cellCentres','r').readlines()
    file = open(case + '/' + str(time) + '/' + var,'r').readlines()
    #lines=0
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    #v= np.zeros([3,70000])
    #cc = False
    j = 0
    for i,line in enumerate(file):
        if 'internalField' in line:
            tmp = i + 1
            tmp2 = i + 3
            #cc = True
            print (tmp, tmp2)
        elif i==tmp:
            maxLines = int(line.split()[0])
            maxIter  = tmp2 + maxLines
            v = np.zeros([1,maxLines])
            print (maxLines, maxIter)
        elif i>=tmp2 and i<maxIter:
            #tmp3=i
            linei = line.split()
            #v[:,j] = [float(linei[0].split('(')[1]), float(linei[1]), float(linei[2].split(')')[0])]
            v[:,j] = [float(linei[0])]
            j += 1
    return v
    
#This function reads the data from a vector RANS file (for example the velocity U) 
#and returns an array v with 3 columns and "maxLines" rows
def getRANSVector(case, time, var):
    #file = open(case + '/0/cellCentres','r').readlines()
    file = open(case + '/' + str(time) + '/' + var,'r').readlines()
    #lines=0
    tmp = [] #The number of data points
    tmp2 = 10**12 
    maxIter = -1
    #v= np.zeros([3,70000])
    #cc = False
    j = 0
    for i,line in enumerate(file):
        if 'internalField' in line:
            tmp = i + 1
            tmp2 = i + 3 
            #cc = True
            #print tmp, tmp2
        elif i==tmp:
            maxLines = int(line.split()[0])
            maxIter  = tmp2 + maxLines
            v = np.zeros([3,maxLines])
           # print maxLines, maxIter
        elif i>=tmp2 and i<maxIter:
            linei = line.split()
            v[:,j] = [float(linei[0].split('(')[1]), float(linei[1]), float(linei[2].split(')')[0])]
            j += 1
    return v

    
#This function reads the data from a vector RANS file (for example the reynolds 
#stress tensor R) and returns a three-dimensional v of "maxLines" two-dimensional arrays 
#with 3 columns and 3 rows
def getRANSTensor(case, time, var):
    file = open(case + '/' + str(time) + '/' + var,'r').readlines()
    #file = open('wavyWall_Re6850_komegaSST_4L_2D/20000/gradU','r').readlines()
    #lines=0
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    #v= np.zeros([3,70000])
    #cc = False
    j = 0
    for i,line in enumerate(file):
        if 'internalField' in line:
            tmp = i + 1
            tmp2 = i + 3
            #cc = True
            print (tmp, tmp2)
        elif i==tmp:
            maxLines = int(line.split()[0])
            maxIter  = tmp2 + maxLines
            v = np.zeros([3,3,maxLines])
            print (maxLines, maxIter)
        elif i>=tmp2 and i<maxIter:
            #tmp3=i
            linei = line.split()
            v[0,:,j] = [float(linei[0].split('(')[1]), float(linei[1]), float(linei[2])]
            v[1,:,j] = [float(linei[3]), float(linei[4]), float(linei[5])]
            v[2,:,j] = [float(linei[6]), float(linei[7]), float(linei[8].split(')')[0])]

            j += 1

    return v
    
#This function reads the data from a vector RANS file (for example the reynolds 
#stress tensor R) and returns a three-dimensional v of "maxLines" two-dimensional arrays 
#with 3 columns and 3 rows  
def getRANSSymmTensor(case, time, var):
    file = open(case + '/' + str(time) + '/' + var,'r').readlines()
    #file = open('wavyWall_Re6850_komegaSST_4L_2D/20000/gradU','r').readlines()
    #lines=0
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    #v= np.zeros([3,70000])
    #cc = False
    j = 0
    for i,line in enumerate(file):
        if 'internalField' in line:
            tmp = i + 1
            tmp2 = i + 3
            #cc = True
            print (tmp, tmp2)
        elif i==tmp:
            maxLines = int(line.split()[0])
            maxIter  = tmp2 + maxLines
            v = np.zeros([3,3,maxLines])
            print( maxLines, maxIter)
        elif i>=tmp2 and i<maxIter:
            #tmp3=i
            linei = line.split()
            v[0,:,j] = [float(linei[0].split('(')[1]), float(linei[1]), float(linei[2])]
            v[1,:,j] = [float(linei[1]), float(linei[3]), float(linei[4])]
            v[2,:,j] = [float(linei[2]), float(linei[4]), float(linei[5].split(')')[0])]

            j += 1

    return v
    
def getRANSPlane(mesh, dimension, nx, ny, t='vector',a='yesAve'):
    xRANS = np.zeros([nx,ny])
    yRANS = np.zeros([nx,ny])
    zRANS = np.zeros([nx,ny])
    print( mesh.shape)

    if a=='yesAve':
        if t=='vector':
            if dimension=='3D':
                for i in range(ny):
                    print ('hello')
#                    xRANS[:,i] = mesh[0,i*256*128:256+i*256*128]
#                    yRANS[:,i] = mesh[1,i*256*128:256+i*256*128]
#                    zRANS[:,i] = mesh[2,i*256*128:256+i*256*128]
            elif dimension=='2D':
                for i in range(ny):
                    xRANS[:,i] = mesh[0,i*nx:nx+i*nx]
                    yRANS[:,i] = mesh[1,i*nx:nx+i*nx]
                    zRANS[:,i] = mesh[2,i*nx:nx+i*nx]

            return np.array([xRANS, yRANS, zRANS])

        elif t=='scalar':
            for i in range(ny):
                xRANS[:,i] = mesh[0,i*nx:nx+i*nx]

            return np.array([xRANS])

        elif t=='tensor':
            out = np.zeros([3,3,nx,ny])
            for i in range(ny):
                out[:,:,:,i] = mesh[:,:,i*nx:nx+i*nx]

            return out

    elif a=='noAve':
        if t=='vector':
            if dimension=='3D':
                for i in range(96):
                    xRANS[:,i] = mesh[0,i*256*128:256+i*256*128]
                    yRANS[:,i] = mesh[1,i*256*128:256+i*256*128]
                    zRANS[:,i] = mesh[2,i*256*128:256+i*256*128]
            elif dimension=='2D':
                for i in range(96):
                    xRANS[:,i] = mesh[0,i*256:256+i*256]
                    yRANS[:,i] = mesh[1,i*256:256+i*256]
                    zRANS[:,i] = mesh[2,i*256:256+i*256]

            return np.array([xRANS, yRANS, zRANS])

def getRANSField(mesh, dimension, nx, ny, nz, t='vector'):
    xRANS = np.zeros([nx,ny*2,nz])
    yRANS = np.zeros([nx,ny*2,nz])
    zRANS = np.zeros([nx,ny*2,nz])
    print (mesh.shape)

    kk=0

    if t=='vector':
        if dimension=='3D':
            print ("3D")
            for j in range(nz):
                for i in range(ny):
                    xRANS[:,i,j] = mesh[0,i*nx+nx*ny*(j+kk*4):nx+i*nx+nx*ny*(j+kk*4)]
                    yRANS[:,i,j] = mesh[1,i*nx+nx*ny*(j+kk*4):nx+i*nx+nx*ny*(j+kk*4)]
                    zRANS[:,i,j] = mesh[2,i*nx+nx*ny*(j+kk*4):nx+i*nx+nx*ny*(j+kk*4)]

        elif dimension=='2D':
            for i in range(ny):
                xRANS[:,i] = mesh[0,i*nx:nx+i*nx]
                yRANS[:,i] = mesh[1,i*nx:nx+i*nx]
                zRANS[:,i] = mesh[2,i*nx:nx+i*nx]

        return np.array([xRANS, yRANS, zRANS])

    elif t=='scalar':
        if dimension=='3D':
            for i in range(ny):
                for j in range(nz):
                    xRANS[:,i,j] = mesh[0,i*nx+nx*ny*j:nx+i*nx+nx*ny*j]
        elif dimension=='2D':
            for i in range(ny):
                xRANS[:,i] = mesh[0,i*nx:nx+i*nx]

        return np.array([xRANS])

    elif t=='tensor':
        out = np.zeros([3,3,nx,ny])
        for i in range(ny):
            out[:,:,:,i] = mesh[:,:,i*nx:nx+i*nx]

        return out
        
def calcInitialConditions(U, turbulenceIntensity, turbLengthScale, nu, d, D):

    k       = 1.5 * (U*turbulenceIntensity)**2.0
    epsilon = 0.16 * k**1.5 / turbLengthScale
    omega   = 1.8 * np.sqrt(k) / turbLengthScale
    #omega_farfield = U/D
    omega_wall = 10 * 6 * nu / (0.0705 * d**2)
    #omega_wall_wilcox = 6 / (0.0708 * yPlus_wilcox**2)
    nuTilda = np.sqrt(1.5)*U*turbulenceIntensity*turbLengthScale
    #nuTilda_NASA = 3*nu
    #nut_NASA = 3*nu*(3**3)/(3**3 + 7.1**3)
    Re      = U*D/nu
    tmp = {'k': k, 'epsilon': epsilon, 'omega': omega, 'nuTilda': nuTilda, 
           'omega_wall':omega_wall, 'Re':Re}    
    
    return tmp

def getSRTensors(gradU,scale=False,k=1.0,eps=1.0):
    # get the strain rate and rotation rate tensors
    # TODO: make exception for non-2D mesh
    a = np.shape(gradU)
    S = np.zeros(a)
    R = np.zeros(a)
    print(a);
    for i1 in range(a[2]):
        for i2 in range(a[3]):               
            #strain rate
            S[:,:,i1,i2] = (0.5)*(gradU[:,:,i1,i2]+np.transpose(gradU[:,:,i1,i2]))
            #rotation rate
            R[:,:,i1,i2] = (0.5)*(gradU[:,:,i1,i2]-np.transpose(gradU[:,:,i1,i2]))
            
            if scale == True:
                S[:,:,i1,i2] = (k[i1,i2]/eps[0,i1,i2])*S[:,:,i1,i2] 
                R[:,:,i1,i2] = (k[i1,i2]/eps[0,i1,i2])*R[:,:,i1,i2] 
    return S,R

    
# Load RANS mesh
#case_dir      = dir_RANS + case_RANS
meshRANSlist  = getRANSVector(dir_RANS, time_end, 'cellCentres')
meshRANS      = getRANSPlane(meshRANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#velocity
U_RANSlist    = getRANSVector(dir_RANS, time_end, 'U')
U_RANS        = getRANSPlane(U_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#velocity gradient
gradU_RANSlist  = getRANSTensor(dir_RANS, time_end, 'grad(U)')
gradU_RANS      = getRANSPlane(gradU_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')

#pressure
p_RANSlist    = getRANSScalar(dir_RANS, time_end, 'p')
p_RANS        = getRANSPlane(p_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')

#pressure gradient
gradp_RANSlist    = getRANSVector(dir_RANS, time_end, 'grad(p)')
gradp_RANS        = getRANSPlane(gradp_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#Reynolds stress tensor
tau_RANSlist  = getRANSSymmTensor(dir_RANS, time_end, 'R')
tau_RANS      = getRANSPlane(tau_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')

#k
k_RANSlist    = getRANSScalar(dir_RANS, time_end, 'k')
k_RANS        = getRANSPlane(k_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')

#k gradient
gradk_RANSlist    = getRANSVector(dir_RANS, time_end, 'grad(k)')
gradk_RANS        = getRANSPlane(gradk_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#distance to wall
yWall_RANSlist = getRANSScalar(dir_RANS, time_end, 'yWall')
yWall_RANS        = getRANSPlane(yWall_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')

#omega
omega_RANSlist  = getRANSScalar(dir_RANS, time_end, 'omega')
omega_RANS      = getRANSPlane(omega_RANSlist, '2D', nx_RANS, ny_RANS, 'scalar')

#S R tensor
S_RANS, Omega_RANS  = getSRTensors(gradU_RANS)

def q1(S_RANS, Omega_RANS): 
    a = np.shape(S_RANS)
    q1 = np.zeros((a[2],a[3]))
    for i1 in range(a[2]):
        for i2 in range(a[3]):               
            raw = 0.5*(np.abs(np.trace(np.dot(S_RANS[:,:,i1,i2],S_RANS[:,:,i1,i2]))) - np.abs(np.trace(np.dot(Omega_RANS[:,:,i1,i2],-1*(Omega_RANS[:,:,i1,i2])))))
            norm = np.trace(np.dot(S_RANS[:,:,i1,i2],S_RANS[:,:,i1,i2]))
            q1[i1,i2] = raw/(np.abs(raw) + np.abs(norm))
    return q1

def q2(k_RANS, U_RANS):
    a = np.shape(k_RANS)
    b= np.shape(U_RANS)
    print( "shape urans=", b)
    q2 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):               
            raw = k_RANS[0,i1,i2]
            norm = 0.5*(np.inner(U_RANS[:, i1, i2], U_RANS[:, i1, i2])) # inner is equivalent to sum UiUi
            q2[i1,i2] = raw/(np.abs(raw) + np.abs(norm))
    return q2



nu=1.4285714285714286e-03

def q3(k_RANS, yWall_RANS, nu):
    a = np.shape(k_RANS)
    q3 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):               
            q3[i1,i2] = np.minimum((np.sqrt(k_RANS[:,i1,i2][0])*yWall_RANS[:, i1, i2])/(50*nu), 2)
    return q3
    
print(q3(k_RANS, yWall_RANS, nu))



def q4(U, gradP):
    a = np.shape(gradP)
    q4 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw  = np.einsum('k,k', U[:,i1,i2], gradP[:,i1,i2])
            norm = np.einsum('j,j,i,i', gradP[:,i1,i2], gradP[:,i1,i2], U[:, i1, i2],U[:, i1, i2])
            
            q4[i1,i2] = raw / (np.fabs(norm) + np.fabs(raw));
    return q4


print(q4(U_RANS, gradp_RANS))

Cmu=0.09
def q5(k_RANS, S_RANS, Cmu, omega_RANS):
    a = np.shape(k_RANS)
    q5 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            epsilon = Cmu * k_RANS[:, i1, i2] * omega_RANS[:, i1, i2]
            raw = k_RANS[:, i1, i2] / epsilon
            norm = 1 / np.sqrt(np.trace(np.dot(S_RANS[:, :, i1, i2],S_RANS[:, :, i1, i2])))
            q5[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q5

print(q5(k_RANS, S_RANS, Cmu, omega_RANS))

def q6(gradP, gradU, p_RANS, U_RANS):
    a = np.shape(gradP)
    q6 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw  = np.sqrt(np.einsum('i,i', gradP[:, i1, i2], gradP[:, i1, i2]))
            norm = np.einsum('k,kk', U_RANS[:, i1, i2], gradU[:, :, i1, i2])
           
            norm *= 0.5 * p_RANS[0, i1, i2]
            q6[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q6
    
print(q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS))


def q7(U_RANS, gradU_RANS):
    a = np.shape(U_RANS)
    q7 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            raw = np.fabs(np.einsum('i, j, ij', U_RANS[:, i1, i2], U_RANS[:, i1, i2], gradU_RANS[:, :, i1, i2]))
            norm = np.sqrt(np.einsum('l, l, i, ij, k, kj', U_RANS[:, i1, i2], U_RANS[:, i1, i2],U_RANS[:, i1, i2], gradU_RANS[:, :, i1, i2], U_RANS[:, i1, i2], gradU_RANS[:, :, i1, i2]))
            q7[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q7


print(q7(U_RANS, gradU_RANS))


def q8(U, gradK, Tau, S):
    a = np.shape(U)
    q8 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw  = np.einsum('i,i',U[:,i1,i2], gradK[:,i1,i2])
            norm = np.einsum('jk,jk', Tau[:,:, i1, i2], S[:,:, i1, i2])
            q8[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))              
    return q8

print (q8(U_RANS,gradk_RANS,tau_RANS,S_RANS))    

def q9(tau_RANS, k_RANS):
    a = np.shape(k_RANS)
    q9 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            raw = np.sqrt(np.trace(np.dot(tau_RANS[:, :, i1, i2],np.transpose(tau_RANS[:, :, i1, i2]))))
            norm = k_RANS[:, i1, i2]
            q9[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q9

print(q9(tau_RANS, k_RANS))

plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], q4(U_RANS, gradp_RANS))
plt.show()

plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS))
plt.show()

plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], q8(U_RANS,gradk_RANS,tau_RANS,S_RANS))
plt.show()



