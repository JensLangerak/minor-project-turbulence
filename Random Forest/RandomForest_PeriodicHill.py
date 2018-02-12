"""
Minor Project

Thomas Stolp

Python code for random forest to predict the Reynods stress discrepancies between RANS and DNS simulations 
of flow over a periodic hill. 

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
            
            phi_RANS =  np.reshape(( np.reshape((foam.eigenvectorToEuler(eigVec_RANS)).swapaxes(1,2), (nx*ny, 3), "F")).swapaxes(1,0), (nx*ny, 3))
            phi_DNS =  np.reshape(( np.reshape((foam.eigenvectorToEuler(eigVec_DNS)).swapaxes(1,2), (nx*ny, 3), "F")).swapaxes(1,0), (nx*ny, 3))
            
            baryMap_discr = np.reshape(( np.reshape((foam.baryMap_discr(baryMap_RANS, baryMap_DNS)).swapaxes(1,2), (nx*ny, 2), "F")).swapaxes(1,0), (nx*ny, 2))

            phi_discr = phi_DNS - phi_RANS
            k_discr = np.reshape(( np.reshape((dataDNS_i['k'] - k_RANS).swapaxes(1,2), (nx*ny, 1), "F")).swapaxes(1,0), (nx*ny, 1))
        
            Y[i*nx*ny:(i+1)*nx*ny, 0:2] = baryMap_discr
            Y[i*nx*ny:(i+1)*nx*ny, 2:5] = phi_discr
            Y[i*nx*ny:(i+1)*nx*ny, 5] = k_discr[:,0]
            

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
            phi_RANS =  foam.eigenvectorToEuler(eigVec_RANS)
            phi_DNS =  foam.eigenvectorToEuler(eigVec_DNS)
            
            phi_discr = phi_DNS - phi_RANS
            k_discr = dataDNS_i['k'] - k_RANS
            print('return bary')
            return baryMap_RANS, baryMap_DNS, baryMap_discr, phi_discr, phi_RANS, k_discr
        
        
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

regr = RandomForestRegressor(n_estimators=40, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features=5, max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, 
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
test_discr = np.reshape(test_discr.swapaxes(1,0), (6, 140, 150))

baryMap_RANS, baryMap_DNS, baryMap_discr, phi_RANS, phi_discr, k_discr = response('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = False)
meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re_test[0], TurbModel, time_end, nx, ny)

test_discr_bary = test_discr[0:2, :, :]
test_discr_phi = test_discr[2:5, :, :]
test_discr_k = test_discr[5:6, :, :]

Y_test = response('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = True)
print(regr.score(test_X, Y_test))

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
print(np.shape(phi_RANS))        



def Ani(baryMap_RANS, baryMap_discr, phi_discr, phi_RANS):
    baryMap_impr = baryMap_RANS + test_discr_bary
    phi_impr = phi_discr + test_discr_phi
    eigenvecs = foam.eulerToEigenvector(phi_impr)
    eigenvals = foam. baryToEigenvals(baryMap_impr)
    #print("shape of eigenvec: ", np.shape(eigenvec))
    #print("shape of eigenvals: ", np.shape(eigenvals))
    #print("shape of k_discr: ", np.shape(k_discr))
    #print("shape of k_RANS: ", np.shape(k_RANS))
    l1 = eigenvecs.shape[2]
    l2 = eigenvecs.shape[3]
    print (l1,l2)
    tauAni = np.zeros([3,3,l1,l2])
    for i in range(l1):
        for j in range(l2):
            tauAni[:,:,i,j] = np.dot(eigenvecs[:,:,i,j], np.dot(np.diag(eigenvals[:, i, j]), np.transpose(eigenvecs[:,:,i,j])))
    return tauAni



###########################################################################################################################
###############################################################################################################################

tauAni = Ani(baryMap_RANS, baryMap_discr, phi_discr, phi_RANS)








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


############################################################################################################
"""
Plots of the barycentric map with coordinates of RANS, DNS and corrected RANS for x = i ... 
"""

i = 50

plt.figure()
#plt.title('DNS %s_Re%i' % (case, Re_test[0]))
plt.plot(baryMap_DNS[0,i,:],baryMap_DNS[1,i,:],'b.', label = 'DNS')
plt.plot(baryMap_DNS[0,i,:],baryMap_DNS[1,i,:])
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.plot(np.add(test_discr[0,i,:], baryMap_RANS[0,i,:]) ,np.add(test_discr[1,i,:],baryMap_RANS[1,i,:]), 'r.', label='Corrected RANS')
plt.plot(np.add(test_discr[0,i,:], baryMap_RANS[0,i,:]) ,np.add(test_discr[1,i,:],baryMap_RANS[1,i,:]))
plt.plot(baryMap_RANS[0,i,:],baryMap_RANS[1,i,:],'k.', label = 'RANS')
plt.plot(baryMap_RANS[0,i,:],baryMap_RANS[1,i,:], 'k-')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.legend()
plt.show()

i = 105

plt.figure()
#plt.title('DNS %s_Re%i' % (case, Re_test[0]))
plt.plot(baryMap_DNS[0,i,:],baryMap_DNS[1,i,:],'b.', label = 'DNS')
plt.plot(baryMap_DNS[0,i,:],baryMap_DNS[1,i,:])
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.plot(np.add(test_discr[0,i,:], baryMap_RANS[0,i,:]) ,np.add(test_discr[1,i,:],baryMap_RANS[1,i,:]), 'r.', label='Corrected RANS')
plt.plot(np.add(test_discr[0,i,:], baryMap_RANS[0,i,:]) ,np.add(test_discr[1,i,:],baryMap_RANS[1,i,:]))
plt.plot(baryMap_RANS[0,i,:],baryMap_RANS[1,i,:],'k.', label = 'RANS')
plt.plot(baryMap_RANS[0,i,:],baryMap_RANS[1,i,:], 'k-')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.legend()
plt.show()


############################################################################################################
"""
Color plot indicating the stress state.
"""

iX, iY = 0,1


def colors_to_cmap(colors):
    """
    Yields a matplotlib colormap object that reproduces the colors in the given
    array when passed a list of N evenly spaced numbers between 0 and 1
    (inclusive), where N is the first dimension of ``colors``.  Allows tripcolor
    plots with arbetrary colors in each triangle.

    Args:
      colors (ndarray (N,[3|4])): RGBa_array
    Return:
      cmap (matplotlib colormap object): Colormap reproducing input colors, 
                                         cmap[i/(N-1)] == colors[i].

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.linspace(0,1,range(len(colors)))

    """
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1), dtype=np.float))/(len(colors) - 1)
    return matplotlib.colors.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] + 
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] + 
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors))


def lena_test():
    """Create a matplotlib tripcolor() plot of an image, with random trigulation."""
    lena = plt.imread("lena.jpg")       # Read image
    h, w, _ = lena.shape
    
    npts = 5000                         # Generate random Delaunay mesh
    pts = np.zeros((npts,2))
    pts[:,0] = np.random.randint(0,h,npts)
    pts[:,1] = np.random.randint(0,w,npts)
    tri = Delaunay(pts)

                                        # Get image colors from triangle centers
    centers = np.sum(pts[tri.simplices]/3, axis=1, dtype='int')
    colors = np.zeros((len(tri.simplices), 3))
    for i,(x,y) in enumerate(centers):
        colors[i,:] = lena[x,y,:]/256.
    cmap = colors_to_cmap(colors)

    plt.xlim(0, w)                      # Do plotting
    plt.ylim(0, h)
    plt.tripcolor(pts[:,1], h-pts[:,0], tri.simplices, 
                  facecolors=np.linspace(0,1,len(tri.simplices)), 
                  edgecolors='none', cmap=cmap) 
    plt.gca().set_aspect('equal')
    plt.show()

    
class BarycentricCoords:
    """
    Mapping between Cartesian/barycentric coordinates on arbetrary triangle.
    """
    def __init__(self):
        # Vertices of output triangle
        self.xv = np.array([[0,0],
                            [1,0],
                            [.5,np.sqrt(3)/2]])
        xv = self.xv
        self.Tinv = np.linalg.inv(
            np.array([ [xv[0,iX]-xv[2,iX], xv[1,iX]-xv[2,iX]],
                       [xv[0,iY]-xv[2,iY], xv[1,iY]-xv[2,iY]] ]))

    def bary2cartesian(self, lam):
        """
        Convert barycentric coordinates (normalized) ``lam`` (ndarray (N,3)), to
        Cartesian coordiates ``x`` (ndarray (N,2)).
        """
        return np.einsum('ij,jk', lam, self.xv)

    def cartesian2bary(self, x):
        """
        Convert Cartesian coordiates ``x`` (ndarray (N,2)), to barycentric 
        coordinates (normalized) ``lam`` (ndarray (N,3)).
        """
        lam = np.zeros((x.shape[0], 3))
        lam[:,:2] = np.einsum('ij,kj->ki', self.Tinv, x - self.xv[2])
        lam[:,2] = 1. - lam[:,0] - lam[:,1]
        return lam

    def trigrid(self, n=10):
        """Uniform grid on triangle in barycentric coordinates."""
        lam = []
        for lam1 in range(n):
            for lam2 in range(n-lam1):
                lam3 = n - lam1 - lam2
                lam.append([lam1, lam2, lam3])
        return np.array(lam)/float(n)
        
    def randomgrid(self, n):
        """Random grid on triangle in barycentric coordinates."""
        lam = np.random.random((n, 3))
        return self.normalize(lam)

    def normalize(self, lam):
        """Normalize Barycentric coordinates to 1."""
        return (lam.T / np.sum(lam, axis=1)).T


def load_breuer_csv():
    """Return coordinates of anisotropy tensor in Barycentric map"""
    import csv
    with open('Hill_Re_10595_Breuer.csv', 'r') as fh:
        reader = csv.reader(fh)
        reader.next()                   # Eat header
        raw = np.array([[float(i) for i in l[:-1]] for l in reader if len(l) > 0])
    return analyse_breuer(raw)

def load_breuer_tri():
    pntfile = 'Hill_Re10595_Breuer_triangulated.points_only.dat'
    trifile = 'Hill_Re10595_Breuer_triangulated.connectivity_only.dat'
    raw  = np.array([float(i) for l in open(pntfile, 'r').readlines()
                              for i in l.strip().split(' ')]).reshape(10, -1).T
    tris = np.array([int(i) for l in open(trifile, 'r').readlines()
                            for i in l.strip().split(' ')], dtype='int').reshape(-1, 3)
    tris -= 1                           # Index from 0
    triangulation = tri.Triangulation(raw[:,0], raw[:,1], triangles=tris)
    return analyse_breuer(raw), triangulation

def analyse_breuer(raw):
    N = raw.shape[0]
    x = raw[:,:2]
    umean = raw[:,2:5]                  # Sensible variable names
    pmean = raw[:,5]
    uu,vv,ww,uv = raw[:,6],raw[:,7],raw[:,8],raw[:,9]
    k = .5 * (uu+vv+ww)
                                        # Anisotropy tensor - where k==0 (e.g. on
                                        # wall) tensor is not defined.
    a = np.zeros((N,3,3))
    a[:,0,0] = uu/(2*k) - 1./3
    a[:,1,1] = vv/(2*k) - 1./3
    a[:,2,2] = ww/(2*k) - 1./3
    a[:,0,1] = a[:,1,0] = uv/(2*k)
    a[k < 1.e-10] = 0.                    
                                        # Eigenvalues
    eigs = np.linalg.eigvalsh(a)
    eigs.sort(axis=1)
    eigs = eigs[:,::-1]
                                        # Barycentric coordinates
    lam = eigs.copy()
    lam[:,0] -= lam[:,1]
    lam[:,1] = 2*(lam[:,1]-lam[:,2])
    lam[:,2] = 3*lam[:,2]+1
    return x, lam, umean


def intersect(A,B,C,D):
    """Return True if line segments AB and CD intersect."""
    def ccw(A,B,C):
        return (C[iY]-A[iY]) * (B[iX]-A[iX]) > (B[iY]-A[iY]) * (C[iX]-A[iX])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def intersection(p1,p2,p3,p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    x = (x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)
    y = (x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)
    p = np.array([x/denom, y/denom])
    xi = np.sqrt(np.sum((p-p1)**2) / np.sum((p2-p1)**2))
    return p, xi

def cut_path(cutpath, mesh):
    """
    Cut a triangulation with a line-segment - return ndarray of edges cut (their 
    indices in the triangulation), and weighting 

    Args:
      cutpath (ndarray (2,2)):
      mesh (tri.Triangulation): The CFD mesh.
    """
    #cutpath = np.array([[x0,0],[x0,2]])
    cutedges = []
    xis, ps = [], []
    for eidx,edge in enumerate(mesh.edges):
        x0 = np.array([mesh.x[edge[0]], mesh.y[edge[0]]])
        x1 = np.array([mesh.x[edge[1]], mesh.y[edge[1]]])
        if intersect(cutpath[0],cutpath[1], x0, x1):
            p, xi = intersection(x0, x1, cutpath[0],cutpath[1])
            xis.append(xi); ps.append(p)
            cutedges.append(eidx)
    cutedges = np.array(cutedges, dtype='int'); ps = np.array(ps)
                                 # Sort according to y-coordinate
    isort = np.argsort(ps[:,1])
    return cutedges[isort], np.array(xis)[isort]


def streamlines(ax, mesh, umean):
    """
    Plot velocity streamlines.  Interpolate data to background Cartesian grid,
    then use matplotlib's ``streamplot()``.
    """
    Nx,Ny = 1001, 401
    xcart = np.linspace(0,10,Nx)
    ycart = np.linspace(0,4,Ny)
    meshcart = np.vstack( map(lambda x: x.flatten(), np.meshgrid(xcart, ycart)) ).T
    interp_u = tri.LinearTriInterpolator(mesh, umean[:,0])
    interp_v = tri.LinearTriInterpolator(mesh, umean[:,1])
    ucart = interp_u(meshcart[:,0], meshcart[:,1])
    vcart = interp_v(meshcart[:,0], meshcart[:,1])
    X = meshcart[:,0].reshape(Ny,Nx)
    Y = meshcart[:,1].reshape(Ny,Nx)
    U = ucart.reshape(Ny,Nx)
    V = vcart.reshape(Ny,Nx)
    lw = np.sqrt(U**2+V**2)*3
    start_points=np.vstack((np.ones(10)*0.1, np.linspace(1,3,10))).T
    ###ax.streamplot(X, Y, U, V, density=.8, color='k', linewidth=1.)
    strm = ax.streamplot(xcart, ycart, U, V, start_points=[[4,2]], color='k', linewidth=1.)
    for path in strm.lines.get_paths():
        path.vertices
    


if __name__ == '__main__':
    barymap = BarycentricCoords()
                                        ### Mesh/data in space
    if True:                            # Triangualtion avaiable
        (xspace, lamspace, umean), trispace = load_breuer_tri()
    else:                               # Only vertex locations available
        xspace, lamspace = load_breuer()
        trispace         = Delaunay(xspace)
                                        ### Mesh for triangular legend
    lamlegend  = barymap.trigrid(100)
    xlegend    = barymap.bary2cartesian(lamlegend)
    trilegend  = Delaunay(xlegend)

                                        ### Build colormaps
    lamcolor = (lamspace.T / np.max(lamspace, axis=1)).T
    lamlegend = (lamlegend.T / np.max(lamlegend, axis=1)).T
    cmap_space = colors_to_cmap(lamcolor)
    cmap_legend = colors_to_cmap(lamlegend)

                                        ### Plotting    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, aspect='equal')
    #fig.add_axes([0,0,1,1])
    ax.tripcolor(xspace[:,0], xspace[:,1], trispace.triangles, 
                 np.linspace(0,1,xspace.shape[0]),
                 edgecolors='none', cmap=cmap_space, shading='gouraud')
    
    #streamlines(ax, trispace, umean)
    #plt.gca().axison = False
    #plt.gca().set_aspect('equal')
    '''
    for j,x0 in enumerate([.5, 1, 1.5, 2, 4, 6, 8, 8.5, 9]):
        print(x0)
        lineseg = np.array([[x0,0],[x0,4]])
        cutedge, xi = cut_path(lineseg, trispace)
        edges = trispace.edges[cutedge]
                                            ### Plot path in space
        xpath = (1-xi) * xspace[edges[:,0]].T + xi*xspace[edges[:,1]].T
        ax.plot(xpath[iX], xpath[iY], '-k')
                                            ### Plot corresponding path in custom
                                            ### barycentric map
                                            # Plot background - same for every path
        axbary = fig.add_axes([.1*j,0.05,.15,.15], aspect='equal')
        axbary.tripcolor(xlegend[:,0], xlegend[:,1], trilegend.simplices,
                         np.linspace(0,1,xlegend.shape[0]),
                         edgecolors='none', cmap=cmap_legend, shading='gouraud')
                                            # Plot path
        lampath = ((1-xi)*lamspace[edges[:,0]].T + xi*lamspace[edges[:,1]].T).T
        cartlegend = barymap.bary2cartesian(lampath[3:-2])
        axbary.plot(cartlegend[:,0], cartlegend[:,1], '-k', linewidth=0.5)
        plt.gca().axison = False
    '''
    x0 = 3
    lineseg = np.array([[x0,0],[x0,4]])
    cutedge, xi = cut_path(lineseg, trispace)
    edges = trispace.edges[cutedge]
                                        ### Plot path in space
    xpath = (1-xi) * xspace[edges[:,0]].T + xi*xspace[edges[:,1]].T
    ax.plot(xpath[iX], xpath[iY], '-k')
    
    x0 = 7
    lineseg = np.array([[x0,0],[x0,4]])
    cutedge, xi = cut_path(lineseg, trispace)
    edges = trispace.edges[cutedge]
                                        ### Plot path in space
    xpath = (1-xi) * xspace[edges[:,0]].T + xi*xspace[edges[:,1]].T
    ax.plot(xpath[iX], xpath[iY], '-k')
                                        ### Plot corresponding path in custom
                                        ### barycentric map
                                        # Plot background - same for every path
    #axbary = fig.add_axes([.1*j,0.05,.15,.15], aspect='equal')
    #axbary.tripcolor(xlegend[:,0], xlegend[:,1], trilegend.simplices,
    #                 np.linspace(0,1,xlegend.shape[0]),
    #                 edgecolors='none', cmap=cmap_legend, shading='gouraud')
                                        # Plot path
    #lampath = ((1-xi)*lamspace[edges[:,0]].T + xi*lamspace[edges[:,1]].T).T
    #cartlegend = barymap.bary2cartesian(lampath[3:-2])
    #axbary.plot(cartlegend[:,0], cartlegend[:,1], '-k', linewidth=0.5)
    plt.gca().axison = False
    plt.show()
    

plt.figure()
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.tripcolor(xlegend[:,0], xlegend[:,1], trilegend.simplices,
                     np.linspace(0,1,xlegend.shape[0]),
                     edgecolors='none', cmap=cmap_legend, shading='gouraud')
plt.axis('equal')
plt.show()


############################################################################################################
"""
Plot of the euclidean norm of the eigenvalue discrepance between the RANS, DNS and corrected RANS
"""


plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(baryMap_RANS,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()

plt.figure()
newBaryMap = np.add(test_discr_bary, baryMap_RANS) 
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(newBaryMap,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()

plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.phi_dist(phi_discr, test_discr_phi),100, cmap = "Reds")
plt.colorbar()
plt.show()















## some other plots
'''
plt.figure()
#plt.title('DNS %s_Re%i' % (case, Re_test[0]))
plt.plot(baryMap_DNS[0,:,:],baryMap_DNS[1,:,:],'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
#plt.title("RANS %s_Re%i corrected with predicted descripancies from RF" % (case, Re_test[0]))
plt.plot(np.add(test_discr[0,:,:], baryMap_RANS[0,:,:]) ,np.add(test_discr[1,:,:],baryMap_RANS[1,:,:]),'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
#plt.title("RANS %s_Re%i" % (case, Re_test[0]))
plt.plot(baryMap_RANS[0,:,:],baryMap_RANS[1,:,:],'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
#plt.title("Eigenvalue discripancy %s_Re%i" % (case, Re_test[0]))
plt.plot(baryMap_discr[0,:,:],baryMap_discr[1,:,:],'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
#plt.title("Discripancy after RF %s_Re%i" % (case, Re_test[0]))
plt.plot(test_discr[0,:,:],test_discr[1,:,:],'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
p#lt.title('DNS %s_Re%i' % (case, Re_test[0]))
plt.plot(baryMap_DNS[0,10,:],baryMap_DNS[1,10,:],'b.')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
#plt.title("RANS %s_Re%i corrected with predicted descripancies from RF" % (case, Re_test[0]))
plt.plot(np.add(test_discr[0,10,:], baryMap_RANS[0,10,:]) ,np.add(test_discr[1,10,:],baryMap_RANS[1,10,:]),'r.')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()


plt.figure()
plt.title("dist") 
contPlot = plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(baryMap_RANS,baryMap_DNS),20,cmap=cmap,extend="both")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.colorbar(contPlot)
plt.figure()


plt.figure()
plt.title("DNS: um")
contPlot = plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:],  dataDNS_i['um'],20,cmap=cmap,extend="both")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.colorbar(contPlot)
plt.show()

plt.figure()
plt.title("RANS: U")
contPlot = plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:],  U_RANS[0,:,:],20,cmap=cmap,extend="both")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.colorbar(contPlot)
plt.show()

plt.figure()
plt.title("DNS: k")
contPlot = plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:],  dataDNS_i['k'],20,cmap=cmap,extend="both")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.colorbar(contPlot)
plt.show()
 
plt.figure()
plt.title("RANS: k") 
contPlot = plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:],  dataRANS_k,20,cmap=cmap,extend="both")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.colorbar(contPlot)
plt.show()


plt.figure()
plt.title("Feature 6")
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS))
plt.show()
'''


##################################################################################################################
####################################### Training parameters ######################################################
##################################################################################################################
'''
def plotRF(Nest, Nfeatures, X, Y, Re_test):
    score = np.zeros((len(Nest),len(Nfeatures)))
    a = np.shape(score)
    for i in range(a[0]):
        for j in range(a[1]):
            regr = RandomForestRegressor(n_estimators=Nest[i], criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                    min_weight_fraction_leaf=0.0, max_features=Nfeatures[j], max_leaf_nodes=None, min_impurity_decrease=0.0, 
                    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, 
                    verbose=0, warm_start=False)
            
            regr.fit(X, Y)
            test_X = features('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150)
            test_discr = regr.predict(test_X)
            test_discr = np.reshape(test_discr.swapaxes(1,0), (6, 140, 150))
            Y_test = response('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = True)
            score[i, j] = regr.score(test_X, Y_test)
    return score
            
Nest = [3, 6, 9, 12, 15, 18, 21]
Nfeatures = [2, 3, 4, 5, 6, 7, 8]

p = plotRF(Nest, Nfeatures, X_train, Y_train, [700])


cmap = 'jet'

plt.figure()
plt.title("score")
plt.contourf(Nfeatures, Nest, p, cmap=cmap)
plt.ylabel("Number of estimators")
plt.xlabel("Number of features")
plt.colorbar()
plt.show()

'''