#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 23:36:11 2018

@author: thomas
"""

"""
Minor Project

Python code for random forest to predict the Reynods stress discrepancies between RANS and DNS simulations 
of flow over a periodic hill. More precisely predicting the eigenvalue discrepancy and the unit quaternions.

Data from:
M. Breuer,(2008), Flow over periodic hills – Numerical and experimental study in a wide range
of Reynolds numbers

Trained on: Re = [700, 1400, 2800, 5600]
Tested on: Re = [10595]
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import openFOAM_FINAL as foam
import os
import sys
sys.path.append("..")
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append("..")
import matplotlib.tri as tri
import matplotlib
from scipy.spatial import Delaunay
from transforms3d.quaternions import quat2mat, mat2quat

def eigenvectorToQuaternion(eigVec):
    q = np.zeros([4 ,eigVec.shape[2],eigVec.shape[3]])
    for i1 in range(eigVec.shape[2]):
        for i2 in range(eigVec.shape[3]):
            quat = mat2quat(eigVec[:,:,i1,i2])
            
            q[0,i1,i2] = quat[0]
            q[1,i1,i2] = quat[1]
            q[2,i1,i2] = quat[2]
            q[3,i1,i2] = quat[3]
    return q

def quaternionToEigenvector(q):
    V = np.zeros([3,3,q.shape[1], q.shape[2]])
    for i1 in range(q.shape[1]):
        for i2 in range(q.shape[2]):
            #rotation matrix:
            a = quat2mat(q[:, i1, i2])
    
            V[:,:,i1,i2] = a
            
    return V




home = os.path.realpath('MinorCSE') + '/'
 
##################################################################################################################
######################################### Loading the RANS data ##################################################
##################################################################################################################
def RANS(case, Re, TurbModel, time_end, nx, ny):
    dir_RANS  = home + ('%s' % case) + '/' + ('Re%i_%s' % (Re,TurbModel))
    if case == 'SquareDuct':
        dir_RANS = dir_RANS + '_50'
        
    mesh_list  = foam.getRANSVector(dir_RANS, time_end, 'cellCentres')
    mesh      = foam.getRANSPlane(mesh_list,'2D', nx, ny, 'vector')
    #velocity
    U_list    = foam.getRANSVector(dir_RANS, time_end, 'U')
    U        = foam.getRANSPlane(U_list,'2D', nx, ny, 'vector')
    #velocity gradient
    gradU_list  = foam.getRANSTensor(dir_RANS, time_end, 'grad(U)')
    gradU      = foam.getRANSPlane(gradU_list,'2D', nx, ny, 'tensor')
    #pressure
    p_list    = foam.getRANSScalar(dir_RANS, time_end, 'p')
    p        = foam.getRANSPlane(p_list,'2D', nx, ny, 'scalar')
    #pressure gradient
    gradp_list    = foam.getRANSVector(dir_RANS, time_end, 'grad(p)')
    gradp        = foam.getRANSPlane(gradp_list,'2D', nx, ny, 'vector')
    #Reynolds stress tensor
    tau_list  = foam.getRANSSymmTensor(dir_RANS, time_end, 'R')
    tau      = foam.getRANSPlane(tau_list,'2D', nx, ny, 'tensor')
    #k
    k_list    = foam.getRANSScalar(dir_RANS, time_end, 'k')
    k        = foam.getRANSPlane(k_list,'2D', nx, ny, 'scalar')
    #k gradient
    gradk_list    = foam.getRANSVector(dir_RANS, time_end, 'grad(k)')
    gradk        = foam.getRANSPlane(gradk_list,'2D', nx, ny, 'vector')
    #distance to wall
    yWall_list = foam.getRANSScalar(dir_RANS, time_end, 'yWall')
    yWall        = foam.getRANSPlane(yWall_list,'2D', nx, ny, 'scalar')
    #omega
    omega_list  = foam.getRANSScalar(dir_RANS, time_end, 'omega')
    omega      = foam.getRANSPlane(omega_list, '2D', nx, ny, 'scalar')
    #S R tensor
    S, Omega  = foam.getSRTensors(gradU)
    
    return mesh, U, gradU, p, gradp, tau, k, gradk, yWall, omega, S, Omega

##################################################################################################################
######################################### Features  ##############################################################
##################################################################################################################
def q1(S, Omega): 
    a = np.shape(S)
    q1 = np.zeros((a[2],a[3]))
    for i1 in range(a[2]):
        for i2 in range(a[3]):               
            raw = 0.5*(np.abs(np.trace(np.dot(S[:,:,i1,i2],S[:,:,i1,i2]))) - np.abs(np.trace(np.dot(Omega[:,:,i1,i2],-1*(Omega[:,:,i1,i2])))))
            norm = np.trace(np.dot(S[:,:,i1,i2],S[:,:,i1,i2]))
            q1[i1,i2] = raw/(np.abs(raw) + np.abs(norm))
    return q1

def q2(k, U):
    a = np.shape(k)
    q2 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):               
            raw = k[0,i1,i2]
            norm = 0.5*(np.inner(U[:, i1, i2], U[:, i1, i2])) # inner is equivalent to sum UiUi
            q2[i1,i2] = raw/(np.abs(raw) + np.abs(norm))
    return q2

    
def q3(k, yWall, nu=1.4285714285714286e-03):
    a = np.shape(k)
    q3 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):               
            q3[i1,i2] = np.minimum((np.sqrt(k[:,i1,i2][0])*yWall[:, i1, i2])/(50*nu), 2)
    return q3
    

def q4(U, gradP):
    a = np.shape(gradP)
    q4 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw  = np.einsum('k,k', U[:,i1,i2], gradP[:,i1,i2])
            norm = np.einsum('j,j,i,i', gradP[:,i1,i2], gradP[:,i1,i2], U[:, i1, i2],U[:, i1, i2])
            
            q4[i1,i2] = raw / (np.fabs(norm) + np.fabs(raw));
    return q4


def q5(k, S, omega, Cmu=0.09):
    a = np.shape(k)
    q5 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            epsilon = Cmu * k[:, i1, i2] * omega[:,i1,i2]
            raw = k[:,i1,i2] / epsilon
            norm = 1 / np.sqrt(np.trace(np.dot(S[:,:,i1,i2], S[:,:,i1,i2])))
            q5[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q5


def q6(gradP, gradU, p, U):
    a = np.shape(gradP)
    q6 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw  = np.sqrt(np.einsum('i,i', gradP[:,i1,i2], gradP[:,i1,i2]))
            norm = np.einsum('k, kk', U[:,i1,i2], gradU[:,:,i1,i2])
           
            norm *= 0.5 * p[0,i1,i2]
            q6[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q6
    

def q7(U, gradU):
    a = np.shape(U)
    q7 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            raw = np.fabs(np.einsum('i, j, ij', U[:,i1,i2], U[:,i1,i2], gradU[:,:,i1,i2]))
            norm = np.sqrt(np.einsum('l, l, i, ij, k, kj', U[:,i1,i2], U[:,i1,i2], U[:,i1,i2], gradU[:,:,i1,i2], U[:,i1,i2], gradU[:,:,i1,i2]))
            q7[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q7



def q8(U, gradK, Tau, S):
    a = np.shape(U)
    q8 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw  = np.einsum('i,i', U[:,i1,i2], gradK[:,i1,i2])
            norm = np.einsum('jk,jk', Tau[:,:,i1,i2], S[:,:,i1,i2])
            q8[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))              
    return q8
  

def q9(tau, k):
    a = np.shape(k)
    q9 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            raw = np.sqrt(np.trace(np.dot(tau[:,:,i1,i2], np.transpose(tau[:,:,i1,i2]))))
            norm = k[:,i1,i2]
            q9[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q9   

##################################################################################################################
######################################### Feature function #######################################################
##################################################################################################################
    
def features(case, Re, TurbModel, time_end, nx, ny):
    X = np.zeros((nx*len(Re) * ny, 9))
    
    for i in range(len(Re)):
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re[i], TurbModel, time_end, nx, ny)
        feature = np.zeros((9, nx, ny))
        feature[0,:,:] = q1(S_RANS, Omega_RANS)
        feature[1,:,:] = q2(k_RANS, U_RANS)
        feature[2,:,:] = q3(k_RANS, yWall_RANS)
        feature[3,:,:] = q4(U_RANS, gradp_RANS)
        feature[4,:,:] = q5(k_RANS, S_RANS, omega_RANS)
        feature[5,:,:] = q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS)
        feature[6,:,:] = q7(U_RANS, gradU_RANS)
        feature[7,:,:] = q8(U_RANS, gradk_RANS, tau_RANS, S_RANS)
        feature[8,:,:] = q9(tau_RANS, k_RANS)
        feature = np.reshape(feature.swapaxes(1,2), (nx*ny, 9), "F")
        feature = np.reshape(feature.swapaxes(1,0), (nx*ny, 9))
        X[i*nx*ny:(i+1)*nx*ny, :] = feature
    return X



##################################################################################################################
##################################### Eigenvalue discripancy function ############################################
##################################################################################################################

def response(case, Re, TurbModel, time_end, nx, ny, train): 
    if train ==True:
        print('train = true')
        Y = np.zeros((nx*len(Re)*ny, 6))
        for i in range(len(Re)):
            if case == 'PeriodicHills':
                dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re[i]) + '/' + ('Hill_Re_%i_Breuer.csv' % Re[i])
            if case == 'SquareDuct':
                dataset = ('MinorCSE/SquareDuct/DATA/0%i_full.csv' % Re[i])
            
            meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re[i], TurbModel, time_end, nx, ny)

            dataDNS = foam.loadData_avg(case, dataset)
            dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
            dataDNS_i['k'] = 0.5 * (dataDNS_i['uu'] + dataDNS_i['vv'] + dataDNS_i['ww'])

            l1 = np.shape(U_RANS)[1]
            l2 = np.shape(U_RANS)[2]

            ReStress_DNS = np.zeros([3,3,l1,l2])
            ReStress_DNS[0,0,:,:] = dataDNS_i['uu']
            ReStress_DNS[1,1,:,:] = dataDNS_i['vv']
            ReStress_DNS[2,2,:,:] = dataDNS_i['ww']
            ReStress_DNS[0,1,:,:] = dataDNS_i['uv']
            ReStress_DNS[1,0,:,:] = dataDNS_i['uv']

            aij_DNS = np.zeros([3,3,l1,l2])
            dataRANS_k = np.zeros([l1,l2])
            dataRANS_aij = np.zeros([3,3,l1,l2])

            for j in range(l1):
                for k in range(l2):
                    aij_DNS[:,:,j,k] = ReStress_DNS[:,:,j,k]/(2.*dataDNS_i['k'][j,k]) - np.diag([1/3.,1/3.,1/3.])
                    dataRANS_k[j,k] = 0.5 * np.trace(tau_RANS[:,:,j,k])
                    dataRANS_aij[:,:,j,k] = tau_RANS[:,:,j,k]/(2.*dataRANS_k[j,k]) - np.diag([1/3.,1/3.,1/3.])

            aneigVal_DNS = foam.calcEigenvalues(ReStress_DNS, dataDNS_i['k'])
            baryMap_DNS = foam.barycentricMap(aneigVal_DNS)

            aneigVal_RANS = foam.calcEigenvalues(tau_RANS, dataRANS_k)
            baryMap_RANS = foam.barycentricMap(aneigVal_RANS)

            
            eigVal_RANS, eigVec_RANS = foam.eigenDecomposition(dataRANS_aij)
            eigVal_DNS, eigVec_DNS = foam.eigenDecomposition(aij_DNS)
            
            q_RANS =  np.reshape(( np.reshape((eigenvectorToQuaternion(eigVec_RANS)).swapaxes(1,2), (nx*ny, 4), "F")).swapaxes(1,0), (nx*ny, 4))
            q_DNS =  np.reshape(( np.reshape((eigenvectorToQuaternion(eigVec_DNS)).swapaxes(1,2), (nx*ny, 4), "F")).swapaxes(1,0), (nx*ny, 4))
            
            baryMap_discr = np.reshape(( np.reshape((foam.baryMap_discr(baryMap_RANS, baryMap_DNS)).swapaxes(1,2), (nx*ny, 2), "F")).swapaxes(1,0), (nx*ny, 2))

            q_discr =  q_DNS 
            k_discr = np.reshape(( np.reshape((dataDNS_i['k'] - k_RANS).swapaxes(1,2), (nx*ny, 1), "F")).swapaxes(1,0), (nx*ny, 1))
        
            Y[i*nx*ny:(i+1)*nx*ny, 0:2] = baryMap_discr
            Y[i*nx*ny:(i+1)*nx*ny, 2:6] = q_discr
            

        print('return Y')
        return Y
    
    else:
        print('train = false')
        for i in range(len(Re)):
            if case == 'PeriodicHills':
                dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re[i]) + '/' + ('Hill_Re_%i_Breuer.csv' % Re[i])
            if case == 'SquareDuct':
                dataset = ('MinorCSE/SquareDuct/DATA/0%i_full.csv' % Re[i])
                
            meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re[i], TurbModel, time_end, nx, ny)

            dataDNS = foam.loadData_avg(case, dataset)
            dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
            dataDNS_i['k'] = 0.5 * (dataDNS_i['uu'] + dataDNS_i['vv'] + dataDNS_i['ww'])

            l1 = np.shape(U_RANS)[1]
            l2 = np.shape(U_RANS)[2]

            ReStress_DNS = np.zeros([3,3,l1,l2])
            ReStress_DNS[0,0,:,:] = dataDNS_i['uu']
            ReStress_DNS[1,1,:,:] = dataDNS_i['vv']
            ReStress_DNS[2,2,:,:] = dataDNS_i['ww']
            ReStress_DNS[0,1,:,:] = dataDNS_i['uv']
            ReStress_DNS[1,0,:,:] = dataDNS_i['uv']

            aij_DNS = np.zeros([3,3,l1,l2])
            dataRANS_k = np.zeros([l1,l2])
            dataRANS_aij = np.zeros([3,3,l1,l2])

            for j in range(l1):
                for k in range(l2):
                    aij_DNS[:,:,j,k] = ReStress_DNS[:,:,j,k]/(2.*dataDNS_i['k'][j,k]) - np.diag([1/3.,1/3.,1/3.])
                    dataRANS_k[j,k] = 0.5 * np.trace(tau_RANS[:,:,j,k])
                    dataRANS_aij[:,:,j,k] = tau_RANS[:,:,j,k]/(2.*dataRANS_k[j,k]) - np.diag([1/3.,1/3.,1/3.])

            aneigVal_DNS = foam.calcEigenvalues(ReStress_DNS, dataDNS_i['k'])
            baryMap_DNS = foam.barycentricMap(aneigVal_DNS)

            aneigVal_RANS = foam.calcEigenvalues(tau_RANS, dataRANS_k)
            baryMap_RANS = foam.barycentricMap(aneigVal_RANS)

            
            eigVal_RANS, eigVec_RANS = foam.eigenDecomposition(dataRANS_aij)
            eigVal_DNS, eigVec_DNS = foam.eigenDecomposition(aij_DNS)
            
            baryMap_discr = foam.baryMap_discr(baryMap_RANS, baryMap_DNS)
            q_RANS =  eigenvectorToQuaternion(eigVec_RANS)
            q_DNS =  eigenvectorToQuaternion(eigVec_DNS)
            
            q_discr = q_DNS
            k_discr = dataDNS_i['k'] - k_RANS
            print('return bary')
            return baryMap_RANS, baryMap_DNS, baryMap_discr, q_discr, q_RANS, k_discr
        
        
##################################################################################################################
##################################################################################################################
######################################### Random forest ##########################################################
##################################################################################################################

case = 'PeriodicHills'

# Training
Re = [700, 1400, 2800, 5600, 10595]
Re_train = np.delete(Re, 4)
nx = 140
ny = 150
time_end = 30000
TurbModel = 'kOmega'
X_train = features('PeriodicHills', Re_train, TurbModel='kOmega', time_end=30000, nx=140, ny=150)

        
'''
case = 'SquareDuct'
Re = [1800, 2000, 2200, 2400, 2600, 2900, 3200, 3500] 
Re_train = np.delete(Re, 2)
TurbModel = 'kOmega'
f = features(case, Re_train, TurbModel='kOmega', time_end=40000, nx=150, ny=150)
X_train = features('SquareDuct', Re_train, TurbModel='kOmega', time_end=40000, nx=150, ny=150)

'''

#######################################################################################################################
#######################################################################################################################
Y_train = response('PeriodicHills', Re_train, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = True)

regr = RandomForestRegressor(n_estimators = 191, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                    min_weight_fraction_leaf=0.0, max_features= 9, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, 
                    verbose=0, warm_start=False)


print()
regr.fit(X_train, Y_train)
print("Feature importance :", regr.feature_importances_) 
print()

# Testing
Re_test = [Re[4]]
print(Re_test)

test_X = features('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150)
test_discr = regr.predict(test_X)

baryMap_RANS, baryMap_DNS, baryMap_discr, q_RANS, q_discr, k_discr = response('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = False)
meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re_test[0], TurbModel, time_end, nx, ny)

Y_test = response('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = True)
print(regr.score(test_X, Y_test))

test_discr = np.reshape(test_discr.swapaxes(1,0), (6, 140, 150))
test_discr_bary = test_discr[0:2, :, :]
test_discr_q = test_discr[2:6, :, :]
#test_discr_k = test_discr[6, :, :]




###########################################################################################################################

Re = [700, 1400, 2800, 5600, 10595]
nx = 140
ny = 150
time_end = 30000
TurbModel = 'kOmega'
dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re[4]) + '/' + ('Hill_Re_%i_Breuer.csv' % Re[4])

dataDNS = foam.loadData_avg(case, dataset)
dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
dataDNS_i['k'] = 0.5 * (dataDNS_i['uu'] + dataDNS_i['vv'] + dataDNS_i['ww'])

l1 = np.shape(U_RANS)[1]
l2 = np.shape(U_RANS)[2]


ReStress_DNS = np.zeros([3,3,l1,l2])

ReStress_DNS[0,0,:,:] = dataDNS_i['uu']
ReStress_DNS[1,1,:,:] = dataDNS_i['vv']
ReStress_DNS[2,2,:,:] = dataDNS_i['ww']
ReStress_DNS[0,1,:,:] = dataDNS_i['uv']
ReStress_DNS[1,0,:,:] = dataDNS_i['uv']

bij_DNS = np.zeros([3,3,l1,l2])
dataRANS_k = np.zeros([l1,l2])
dataRANS_bij = np.zeros([3,3,l1,l2])

for i in range(l1):
    for j in range(l2):
        bij_DNS[:,:,i,j] = ReStress_DNS[:,:,i,j]/(2.*dataDNS_i['k'][i,j]) - np.diag([1/3.,1/3.,1/3.])
        dataRANS_k[i,j] = 0.5 * np.trace(tau_RANS[:,:,i,j])
        dataRANS_bij[:,:,i,j] = tau_RANS[:,:,i,j]/(2.*dataRANS_k[i,j]) - np.diag([1/3.,1/3.,1/3.])
        

def Ani(baryMap_RANS, test_discr_bary, test_discr_q, q_RANS):
    baryMap_impr = baryMap_RANS + test_discr_bary
    q_impr = test_discr_q

    eigenvecs = quaternionToEigenvector(q_impr)
    eigenvals = foam.baryToEigenvals(baryMap_impr)
    l1 = eigenvecs.shape[2]
    l2 = eigenvecs.shape[3]
    print (l1,l2)
    tauAni = np.zeros([3,3,l1,l2])
    for i in range(l1):
        for j in range(l2):
            tauAni[:,:,i,j] = (np.dot(eigenvecs[:,:,i,j], np.dot(np.diag(eigenvals[:, i, j]), np.transpose(eigenvecs[:,:,i,j]))))
            
    return tauAni



###########################################################################################################################
###############################################################################################################################

tauAni = Ani(baryMap_RANS, test_discr_bary, test_discr_q, q_RANS)




############################################################################################################
"""
Plot of the nonzero components of the Reynolds stress anisotropy tensor.
"""

# Plots
cmap=plt.cm.coolwarm
cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,10))

im = axes[0,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], bij_DNS[0,0,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[0,0].set_title('DNS $b_{11}$')
axes[0,0].set_title('DNS')
axes[0,0].set(ylabel='$b_{11}$')
fig.colorbar(im, ax = axes[0,0])

im =axes[0,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS_bij[0,0,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
axes[0,1].set_title('RANS')
#axes[0,1].set_title('RANS $b_{11}$')
fig.colorbar(im, ax = axes[0,1])


im = axes[0,2].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tauAni[0,0,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
plt.xlabel('x-axis')
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
axes[0,2].set_title('ML')
#axes[0,2].set_title('ML $b_{11}$')
fig.colorbar(im, ax = axes[0,2])

im = axes[1,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], bij_DNS[0,1,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,0].set_title('DNS $b_{12}$')
axes[1,0].set(ylabel='$b_{12}$')
fig.colorbar(im, ax = axes[1,0])

im = axes[1,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:],  dataRANS_bij[0,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,1].set_title('RANS $b_{12}$')
fig.colorbar(im, ax = axes[1,1])

im =axes[1,2].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tauAni[0,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,2].set_title('ML $b_{12}$')
fig.colorbar(im, ax = axes[1,2])


im = axes[2,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], bij_DNS[1,1,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[3,0].set_title('DNS $b_{22}$')
axes[2,0].set(ylabel='$b_{22}$')
fig.colorbar(im, ax = axes[2,0])

im = axes[2,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS_bij[1,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,1].set_title('RANS $b_{22}$')
fig.colorbar(im, ax = axes[2,1])

im =axes[2,2].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tauAni[1,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[3,2].set_title('ML $b_{22}$')
fig.colorbar(im, ax = axes[2,2])

im = axes[3,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], bij_DNS[2,2,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[5,0].set_title('DNS $b_{33}$')
axes[3,0].set(ylabel='$b_{33}$')
fig.colorbar(im, ax = axes[3,0])


im = axes[3,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS_bij[2,2,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[5,1].set_title('RANS $b_{33}$')
fig.colorbar(im, ax = axes[3,1])

im = axes[3,2].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tauAni[2,2,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[5,2].set_title('ML $b_{33}$')
fig.colorbar(im, ax = axes[3,2])

fig.tight_layout()
plt.show()










