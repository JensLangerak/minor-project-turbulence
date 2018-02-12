"""
Minor Project

Thomas Stolp

Python code for random forest to predict the Reynods stress discrepancies between RANS and DNS simulations 
of flow in a converging diverging channel. 

Data from:
M. Breuer,(2008), Flow over periodic hills – Numerical and experimental study in a wide range
of Reynolds numbers



Trained on: 
    Periodic hill: Re = [700, 1400, 2800, 5600 ,10595]
    Square duct: Re = [1800, 2000, 2200, 2400, 2600, 2900, 3200, 3500]
Tested on: 
    Converging diverging channel: Re = [12600]
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

home = os.path.realpath('MinorCSE') + '/'




##################################################################################################################
######################################### Loading the RANS data ##################################################
##################################################################################################################
def RANS(case, Re, TurbModel, time_end, nx, ny):
    dir_RANS  = home + ('%s' % case) + '/' + ('Re%i_%s' % (Re,TurbModel))
    if case == 'SquareDuct':
        dir_RANS = dir_RANS + '_50'
        if Re > 2000:
            time_end = 50000
            
    if case == 'ConvergingDivergingChannel':
        dir_RANS = dir_RANS + '_100'
        
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
            if case == 'ConvergingDivergingChannel':
                dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
                
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
        
        
def responseCD(case, Re, TurbModel, time_end, nx, ny, train): 
    if train ==True:
        print('train = true')
        Y = np.zeros((nx*len(Re)*ny, 6))
        
            
        if case == 'ConvergingDivergingChannel':
            dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
        else:
            print('wrong flowcase!')
       
            
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re[0], TurbModel, time_end, nx, ny)

        dataDNS = foam.loadData_avg(case, dataset)
        dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
        dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

        l1 = np.shape(U_RANS)[1]
        l2 = np.shape(U_RANS)[2]

        ReStress_DNS = np.zeros([3,3,l1,l2])
        ReStress_DNS[0,0,:,:] = dataDNS_i[14]
        ReStress_DNS[1,1,:,:] = dataDNS_i[17]
        ReStress_DNS[2,2,:,:] = dataDNS_i[19]
        ReStress_DNS[0,1,:,:] = dataDNS_i[15]
        ReStress_DNS[1,0,:,:] = dataDNS_i[15]

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
    
        Y[0:nx*ny, 0:2] = baryMap_discr
        Y[0:nx*ny, 2:5] = phi_discr
        Y[0:nx*ny, 5] = k_discr[:,0]
            

        print('return Y')
        return Y
    
    else:
        print('train = false')
        
        if case == 'PeriodicHills':
            dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re) + '/' + ('Hill_Re_%i_Breuer.csv' % Re)
        if case == 'SquareDuct':
            dataset = ('MinorCSE/SquareDuct/DATA/0%i_full.csv' % Re)
        if case == 'ConvergingDivergingChannel':
            dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
            
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re, TurbModel, time_end, nx, ny)

        dataDNS = foam.loadData_avg(case, dataset)
        dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
        dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

        l1 = np.shape(U_RANS)[1]
        l2 = np.shape(U_RANS)[2]

        ReStress_DNS = np.zeros([3,3,l1,l2])
        ReStress_DNS[0,0,:,:] = dataDNS_i[14]
        ReStress_DNS[1,1,:,:] = dataDNS_i[17]
        ReStress_DNS[2,2,:,:] = dataDNS_i[19]
        ReStress_DNS[0,1,:,:] = dataDNS_i[15]
        ReStress_DNS[1,0,:,:] = dataDNS_i[15]

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
        


################################################## TRAINING #########################################################################



#case = 'PeriodicHills'
Re1 = [700, 1400, 2800, 5600, 10595]
Re_train1 = Re1
#Re_train1 = np.delete(Re1, 0)
#TurbModel = 'kOmega'
X_train1 = features('PeriodicHills', Re_train1, TurbModel='kOmega', time_end=30000, nx=140, ny=150)


#case = 'SquareDuct'
Re2 = [1800, 2000, 2200, 2400, 2600, 2900, 3200, 3500] 
Re_train2 = Re2
#Re_train2 = np.delete(Re2, 2)
#TurbModel = 'kOmega'
X_train2 = features('SquareDuct', Re_train2, TurbModel='kOmega', time_end=40000, nx=50, ny=50)


def mergedata(X_train1, X_train2):
    a = np.shape(X_train1)
    b = np.shape(X_train2)
    X = np.zeros((a[0]+b[0], a[1]))
    for i in range(a[0]):
        for j in range(a[1]):
            X[i, j] = X_train1[i, j]
    for k in range(b[0]):
        for l in range(a[1]):
            X[k+a[0], l] = X_train2[k, l]
    return X

X_train = mergedata(X_train1, X_train2)
#X_train = features('ConvergingDivergingChannel', Re_train, TurbModel='kOmega', time_end=7000, nx=140, ny=100)



Y_train1 = response('PeriodicHills', Re_train1, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = True)
Y_train2 = response('SquareDuct', Re_train2, TurbModel='kOmega', time_end=40000, nx=50, ny=50, train = True)

Y_train = mergedata(Y_train1, Y_train2)

regr = RandomForestRegressor(n_estimators=12, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features= 3, max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, 
    verbose=0, warm_start=False)

regr.fit(X_train, Y_train)



################################################ TESTING #################################################################

case = 'ConvergingDivergingChannel'
Re_test = [12600]
TurbModel = 'kOmega'
time_end = 7000
nx=140
ny=100
#dir_RANS  = home + ('%s' % case) + '/' + ('Re%i_%s_100' % (Re_test,TurbModel))
X_test = features('ConvergingDivergingChannel', Re_test, TurbModel='kOmega', time_end=7000, nx=140, ny=100)
print(np.shape(X_test))
test_discr = regr.predict(X_test)
test_discr = np.reshape(test_discr.swapaxes(1,0), (6, 140, 100))



baryMap_RANS, baryMap_DNS, baryMap_discr, phi_RANS, phi_discr, k_discr = responseCD('ConvergingDivergingChannel', Re_test[0], TurbModel='kOmega', time_end=7000, nx=140, ny=100, train = False)
meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re_test[0], TurbModel, time_end, nx, ny)

Y_test = responseCD('ConvergingDivergingChannel', Re_test, TurbModel='kOmega', time_end=7000, nx=140, ny=100, train = True)
print("Feature importance :", regr.feature_importances_) 
print("Train for eigenvalues and euler angles at the same time ... ")
print('score :', regr.score(X_test, Y_test))
print()

# Plots

test_discr_bary = test_discr[0:2, :, :]
test_discr_phi = test_discr[2:5, :, :]
print(np.shape(test_discr_phi))

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

case = 'ConvergingDivergingChannel'
Re_test = [12600]
TurbModel = 'kOmega'
time_end = 7000
nx=140
ny=100

dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')

dataDNS = foam.loadData_avg(case, dataset)
dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

l1 = np.shape(U_RANS)[1]
l2 = np.shape(U_RANS)[2]

ReStress_DNS = np.zeros([3,3,l1,l2])
ReStress_DNS[0,0,:,:] = dataDNS_i[14]
ReStress_DNS[1,1,:,:] = dataDNS_i[17]
ReStress_DNS[2,2,:,:] = dataDNS_i[19]
ReStress_DNS[0,1,:,:] = dataDNS_i[15]
ReStress_DNS[1,0,:,:] = dataDNS_i[15]

bij_DNS = np.zeros([3,3,l1,l2])
dataRANS_k = np.zeros([l1,l2])
dataRANS_bij = np.zeros([3,3,l1,l2])

for j in range(l1):
    for k in range(l2):
        bij_DNS[:,:,j,k] = ReStress_DNS[:,:,j,k]/(2.*dataDNS_i['k'][j,k]) - np.diag([1/3.,1/3.,1/3.])
        dataRANS_k[j,k] = 0.5 * np.trace(tau_RANS[:,:,j,k])
        dataRANS_bij[:,:,j,k] = tau_RANS[:,:,j,k]/(2.*dataRANS_k[j,k]) - np.diag([1/3.,1/3.,1/3.])

tauAni = Ani(baryMap_RANS, baryMap_discr, phi_discr, phi_RANS)

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

im =axes[0,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:],  dataRANS_bij[0,0,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
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

im = axes[2,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:],  dataRANS_bij[1,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
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


im = axes[3,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:],  dataRANS_bij[2,2,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
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
'''
plt.figure()
newBaryMap = np.add(test_discr_bary, baryMap_RANS) 
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(newBaryMap,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()


plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(baryMap_RANS,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()

print('Difference of the two above plots')
plt.figure()
newBaryMap = np.add(test_discr_bary, baryMap_RANS) 
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], np.add(foam.baryMap_dist(baryMap_RANS,baryMap_DNS), -foam.baryMap_dist(newBaryMap,baryMap_DNS)),100, cmap = "Reds")
plt.colorbar()
plt.show()


plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.phi_dist(test_discr, test_discr_phi),100, cmap = "Reds")
plt.colorbar()
plt.show()


def response_eig(case, Re, TurbModel, time_end, nx, ny, train): 
    if train ==True:
        print('train = true')
        Y = np.zeros((nx*len(Re)*ny, 2))
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
            
            #hi_RANS =  np.reshape(( np.reshape((foam.eigenvectorToEuler(eigVec_RANS)).swapaxes(1,2), (nx*ny, 3), "F")).swapaxes(1,0), (nx*ny, 3))
            #phi_DNS =  np.reshape(( np.reshape((foam.eigenvectorToEuler(eigVec_DNS)).swapaxes(1,2), (nx*ny, 3), "F")).swapaxes(1,0), (nx*ny, 3))
            
            baryMap_discr = np.reshape(( np.reshape((foam.baryMap_discr(baryMap_RANS, baryMap_DNS)).swapaxes(1,2), (nx*ny, 2), "F")).swapaxes(1,0), (nx*ny, 2))

            #phi_discr = phi_DNS - phi_RANS
            #k_discr = np.reshape(( np.reshape((dataDNS_i['k'] - k_RANS).swapaxes(1,2), (nx*ny, 1), "F")).swapaxes(1,0), (nx*ny, 1))
        
            Y[i*nx*ny:(i+1)*nx*ny, 0:2] = baryMap_discr
            #Y[i*nx*ny:(i+1)*nx*ny, 2:5] = phi_discr
            #Y[i*nx*ny:(i+1)*nx*ny, 5] = k_discr[:,0]
            

        print('return Y')
        return Y
    
    else:
        print('train = false')
        for i in range(len(Re)):
            if case == 'PeriodicHills':
                dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re[i]) + '/' + ('Hill_Re_%i_Breuer.csv' % Re[i])
            if case == 'SquareDuct':
                dataset = ('MinorCSE/SquareDuct/DATA/0%i_full.csv' % Re[i])
            if case == 'ConvergingDivergingChannel':
                dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
                
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
            #phi_RANS =  foam.eigenvectorToEuler(eigVec_RANS)
            #phi_DNS =  foam.eigenvectorToEuler(eigVec_DNS)
            
           # phi_discr = phi_DNS - phi_RANS
           # k_discr = dataDNS_i['k'] - k_RANS
            print('return bary')
            return baryMap_RANS, baryMap_DNS, baryMap_discr #, phi_discr, phi_RANS, k_discr
        
        
def responseCD_eig(case, Re, TurbModel, time_end, nx, ny, train): 
    if train ==True:
        print('train = true')
        Y = np.zeros((nx*len(Re)*ny, 2))
        
            
        if case == 'ConvergingDivergingChannel':
            dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
        else:
            print('wrong flowcase!')
       
            
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re[0], TurbModel, time_end, nx, ny)

        dataDNS = foam.loadData_avg(case, dataset)
        dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
        dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

        l1 = np.shape(U_RANS)[1]
        l2 = np.shape(U_RANS)[2]

        ReStress_DNS = np.zeros([3,3,l1,l2])
        ReStress_DNS[0,0,:,:] = dataDNS_i[14]
        ReStress_DNS[1,1,:,:] = dataDNS_i[17]
        ReStress_DNS[2,2,:,:] = dataDNS_i[19]
        ReStress_DNS[0,1,:,:] = dataDNS_i[15]
        ReStress_DNS[1,0,:,:] = dataDNS_i[15]

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
        
        #phi_RANS =  np.reshape(( np.reshape((foam.eigenvectorToEuler(eigVec_RANS)).swapaxes(1,2), (nx*ny, 3), "F")).swapaxes(1,0), (nx*ny, 3))
        #phi_DNS =  np.reshape(( np.reshape((foam.eigenvectorToEuler(eigVec_DNS)).swapaxes(1,2), (nx*ny, 3), "F")).swapaxes(1,0), (nx*ny, 3))
        
        baryMap_discr = np.reshape(( np.reshape((foam.baryMap_discr(baryMap_RANS, baryMap_DNS)).swapaxes(1,2), (nx*ny, 2), "F")).swapaxes(1,0), (nx*ny, 2))

        #phi_discr = phi_DNS - phi_RANS
        #k_discr = np.reshape(( np.reshape((dataDNS_i['k'] - k_RANS).swapaxes(1,2), (nx*ny, 1), "F")).swapaxes(1,0), (nx*ny, 1))
    
        Y[0:nx*ny, 0:2] = baryMap_discr
        #Y[0:nx*ny, 2:5] = phi_discr
        #Y[0:nx*ny, 5] = k_discr[:,0]
            

        print('return Y')
        return Y
    
    else:
        print('train = false')
        
        if case == 'PeriodicHills':
            dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re) + '/' + ('Hill_Re_%i_Breuer.csv' % Re)
        if case == 'SquareDuct':
            dataset = ('MinorCSE/SquareDuct/DATA/0%i_full.csv' % Re)
        if case == 'ConvergingDivergingChannel':
            dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
            
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re, TurbModel, time_end, nx, ny)

        dataDNS = foam.loadData_avg(case, dataset)
        dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
        dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

        l1 = np.shape(U_RANS)[1]
        l2 = np.shape(U_RANS)[2]

        ReStress_DNS = np.zeros([3,3,l1,l2])
        ReStress_DNS[0,0,:,:] = dataDNS_i[14]
        ReStress_DNS[1,1,:,:] = dataDNS_i[17]
        ReStress_DNS[2,2,:,:] = dataDNS_i[19]
        ReStress_DNS[0,1,:,:] = dataDNS_i[15]
        ReStress_DNS[1,0,:,:] = dataDNS_i[15]

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
        #phi_RANS =  foam.eigenvectorToEuler(eigVec_RANS)
        #phi_DNS =  foam.eigenvectorToEuler(eigVec_DNS)

        #phi_discr = phi_DNS - phi_RANS
        #k_discr = dataDNS_i['k'] - k_RANS
        print('return bary')
        return baryMap_RANS, baryMap_DNS, baryMap_discr #, phi_discr, phi_RANS, k_discr
        

def response_phi(case, Re, TurbModel, time_end, nx, ny, train): 
    if train ==True:
        print('train = true')
        Y = np.zeros((nx*len(Re)*ny, 3))
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
            #k_discr = np.reshape(( np.reshape((dataDNS_i['k'] - k_RANS).swapaxes(1,2), (nx*ny, 1), "F")).swapaxes(1,0), (nx*ny, 1))
        
            #Y[i*nx*ny:(i+1)*nx*ny, 0:2] = baryMap_discr
            Y[i*nx*ny:(i+1)*nx*ny, 0:3] = phi_discr
            #Y[i*nx*ny:(i+1)*nx*ny, 5] = k_discr[:,0]
            

        print('return Y')
        return Y
    
    else:
        print('train = false')
        for i in range(len(Re)):
            if case == 'PeriodicHills':
                dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re[i]) + '/' + ('Hill_Re_%i_Breuer.csv' % Re[i])
            if case == 'SquareDuct':
                dataset = ('MinorCSE/SquareDuct/DATA/0%i_full.csv' % Re[i])
            if case == 'ConvergingDivergingChannel':
                dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
                
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
            #k_discr = dataDNS_i['k'] - k_RANS
            print('return bary')
            return phi_discr, phi_RANS
        
        
def responseCD_phi(case, Re, TurbModel, time_end, nx, ny, train): 
    if train ==True:
        print('train = true')
        Y = np.zeros((nx*len(Re)*ny, 3))
        
            
        if case == 'ConvergingDivergingChannel':
            dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
        else:
            print('wrong flowcase!')
       
            
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re[0], TurbModel, time_end, nx, ny)

        dataDNS = foam.loadData_avg(case, dataset)
        dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
        dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

        l1 = np.shape(U_RANS)[1]
        l2 = np.shape(U_RANS)[2]

        ReStress_DNS = np.zeros([3,3,l1,l2])
        ReStress_DNS[0,0,:,:] = dataDNS_i[14]
        ReStress_DNS[1,1,:,:] = dataDNS_i[17]
        ReStress_DNS[2,2,:,:] = dataDNS_i[19]
        ReStress_DNS[0,1,:,:] = dataDNS_i[15]
        ReStress_DNS[1,0,:,:] = dataDNS_i[15]

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
        #k_discr = np.reshape(( np.reshape((dataDNS_i['k'] - k_RANS).swapaxes(1,2), (nx*ny, 1), "F")).swapaxes(1,0), (nx*ny, 1))
    
        #Y[0:nx*ny, 0:2] = baryMap_discr
        Y[0:nx*ny, 0:3] = phi_discr
        #Y[0:nx*ny, 5] = k_discr[:,0]
            

        print('return Y')
        return Y
    
    else:
        print('train = false')
        
        if case == 'PeriodicHills':
            dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re) + '/' + ('Hill_Re_%i_Breuer.csv' % Re)
        if case == 'SquareDuct':
            dataset = ('MinorCSE/SquareDuct/DATA/0%i_full.csv' % Re)
        if case == 'ConvergingDivergingChannel':
            dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')
            
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re, TurbModel, time_end, nx, ny)

        dataDNS = foam.loadData_avg(case, dataset)
        dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
        dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

        l1 = np.shape(U_RANS)[1]
        l2 = np.shape(U_RANS)[2]

        ReStress_DNS = np.zeros([3,3,l1,l2])
        ReStress_DNS[0,0,:,:] = dataDNS_i[14]
        ReStress_DNS[1,1,:,:] = dataDNS_i[17]
        ReStress_DNS[2,2,:,:] = dataDNS_i[19]
        ReStress_DNS[0,1,:,:] = dataDNS_i[15]
        ReStress_DNS[1,0,:,:] = dataDNS_i[15]

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
        #k_discr = dataDNS_i['k'] - k_RANS
        print('return bary')
        return phi_discr, phi_RANS
        
################################################## TRAINING #########################################################################


## calculating features 

#case = 'PeriodicHills'
Re1 = [700, 1400, 2800, 5600, 10595]
Re_train1 = Re1
#Re_train1 = np.delete(Re1, 0)
#TurbModel = 'kOmega'
X_train1 = features('PeriodicHills', Re_train1, TurbModel='kOmega', time_end=30000, nx=140, ny=150)


#case = 'SquareDuct'
Re2 = [1800, 2000, 2200, 2400, 2600, 2900, 3200, 3500] 
Re_train2 = Re2
#Re_train2 = np.delete(Re2, 2)
#TurbModel = 'kOmega'
X_train2 = features('SquareDuct', Re_train2, TurbModel='kOmega', time_end=40000, nx=50, ny=50)



def mergedata(X_train1, X_train2):
    '''
    Funtion to merge arrays of features from different flowcases
    '''
    a = np.shape(X_train1)
    b = np.shape(X_train2)
    X = np.zeros((a[0]+b[0], a[1]))
    for i in range(a[0]):
        for j in range(a[1]):
            X[i, j] = X_train1[i, j]
    for k in range(b[0]):
        for l in range(a[1]):
            X[k+a[0], l] = X_train2[k, l]
    return X

X_train = mergedata(X_train1, X_train2)
#X_train = features('ConvergingDivergingChannel', Re_train, TurbModel='kOmega', time_end=7000, nx=140, ny=100)


################################################################# bary eigenvalues ################################

Y_train1_eig = response_eig('PeriodicHills', Re_train1, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = True)
Y_train2_eig = response_eig('SquareDuct', Re_train2, TurbModel='kOmega', time_end=40000, nx=50, ny=50, train = True)

Y_train_eig = mergedata(Y_train1_eig, Y_train2_eig)

regr = RandomForestRegressor(n_estimators=9, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features= 3, max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, 
    verbose=0, warm_start=False)

regr.fit(X_train, Y_train_eig)
print("Feature importance :", regr.feature_importances_) 


case = 'ConvergingDivergingChannel'
Re_test = [12600]
TurbModel = 'kOmega'
time_end = 7000
nx=140
ny=100
#dir_RANS  = home + ('%s' % case) + '/' + ('Re%i_%s_100' % (Re_test,TurbModel))
X_test = features('ConvergingDivergingChannel', Re_test, TurbModel='kOmega', time_end=7000, nx=140, ny=100)
print(np.shape(X_test))


test_discr = regr.predict(X_test)
test_discr = np.reshape(test_discr.swapaxes(1,0), (2, 140, 100))



baryMap_RANS, baryMap_DNS, baryMap_discr  = responseCD_eig('ConvergingDivergingChannel', Re_test[0], TurbModel='kOmega', time_end=7000, nx=140, ny=100, train = False)
meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re_test[0], TurbModel, time_end, nx, ny)


test_discr_bary = test_discr[0:2, :, :]

Y_test = responseCD_eig('ConvergingDivergingChannel', Re_test, TurbModel='kOmega', time_end=7000, nx=140, ny=100, train = True)
print("Feature importance :", regr.feature_importances_) 
print("Train for eigenvalues only ")
print('score :', regr.score(X_test, Y_test))
print()

plt.figure()
#plt.title('DNS %s_Re%i' % (case, Re_test[0]))
plt.plot(baryMap_DNS[0,:,:],baryMap_DNS[1,:,:],'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
#plt.title("RANS %s_Re%i corrected with predicted descripancies from RF" % (case, Re_test[0]))
plt.plot(np.add(test_discr_bary[0,:,:], baryMap_RANS[0,:,:]) ,np.add(test_discr_bary[1,:,:],baryMap_RANS[1,:,:]),'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

plt.figure()
#plt.title("RANS %s_Re%i" % (case, Re_test[0]))
plt.plot(baryMap_RANS[0,:,:],baryMap_RANS[1,:,:],'b*')
plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
plt.axis('equal')
plt.show()

############################################################################ phi euler angles ############################

Y_train1_phi = response_phi('PeriodicHills', Re_train1, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train = True)
Y_train2_phi = response_phi('SquareDuct', Re_train2, TurbModel='kOmega', time_end=40000, nx=50, ny=50, train = True)

Y_train_phi = mergedata(Y_train1_phi, Y_train2_phi)

regr = RandomForestRegressor(n_estimators=22, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features= 3, max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, 
    verbose=0, warm_start=False)

regr.fit(X_train, Y_train_phi)
print("Feature importance :", regr.feature_importances_)

case = 'ConvergingDivergingChannel'
Re_test = [12600]
TurbModel = 'kOmega'
time_end = 7000
nx=140
ny=100
#dir_RANS  = home + ('%s' % case) + '/' + ('Re%i_%s_100' % (Re_test,TurbModel))
X_test = features('ConvergingDivergingChannel', Re_test, TurbModel='kOmega', time_end=7000, nx=140, ny=100)


test_discr = regr.predict(X_test)
test_discr = np.reshape(test_discr.swapaxes(1,0), (3, 140, 100))
print()

phi_discr, phi_RANS = responseCD_phi('ConvergingDivergingChannel', Re_test[0], TurbModel='kOmega', time_end=7000, nx=140, ny=100, train = False)
meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(case, Re_test[0], TurbModel, time_end, nx, ny)


test_discr_phi = test_discr[0:3, :, :]

Y_test = responseCD_phi('ConvergingDivergingChannel', Re_test, TurbModel='kOmega', time_end=7000, nx=140, ny=100, train = True)
print("Feature importance :", regr.feature_importances_) 
print("Train for phi only ")
print('score :', regr.score(X_test, Y_test))
print()

def Ani(baryMap_RANS, test_discr_bary, test_phi_discr, phi_RANS):
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

case = 'ConvergingDivergingChannel'
Re_test = [12600]
TurbModel = 'kOmega'
time_end = 7000
nx=140
ny=100

dataset = ('MinorCSE/ConvergingDivergingChannel/DATA/conv-div-mean-half.dat')

dataDNS = foam.loadData_avg(case, dataset)
dataDNS_i = foam.interpDNSOnRANS(case, dataDNS, meshRANS)
dataDNS_i['k'] = 0.5 * (dataDNS_i[14] + dataDNS_i[17] + dataDNS_i[19])

l1 = np.shape(U_RANS)[1]
l2 = np.shape(U_RANS)[2]

ReStress_DNS = np.zeros([3,3,l1,l2])
ReStress_DNS[0,0,:,:] = dataDNS_i[14]
ReStress_DNS[1,1,:,:] = dataDNS_i[17]
ReStress_DNS[2,2,:,:] = dataDNS_i[19]
ReStress_DNS[0,1,:,:] = dataDNS_i[15]
ReStress_DNS[1,0,:,:] = dataDNS_i[15]

aij_DNS = np.zeros([3,3,l1,l2])
dataRANS_k = np.zeros([l1,l2])
dataRANS_aij = np.zeros([3,3,l1,l2])

for j in range(l1):
    for k in range(l2):
        aij_DNS[:,:,j,k] = ReStress_DNS[:,:,j,k]/(2.*dataDNS_i['k'][j,k]) - np.diag([1/3.,1/3.,1/3.])
        dataRANS_k[j,k] = 0.5 * np.trace(tau_RANS[:,:,j,k])
        dataRANS_aij[:,:,j,k] = tau_RANS[:,:,j,k]/(2.*dataRANS_k[j,k]) - np.diag([1/3.,1/3.,1/3.])

tauAni = Ani(baryMap_RANS, test_discr_bary, test_discr_phi, phi_RANS)

cmap=plt.cm.coolwarm
cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,10))

im = axes[0,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], aij_DNS[0,0,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[0,0].set_title('DNS $b_{11}$')
axes[0,0].set_title('DNS')
axes[0,0].set(ylabel='$b_{11}$')
fig.colorbar(im, ax = axes[0,0])

im =axes[0,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tau_RANS[0,0,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
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

im = axes[1,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], aij_DNS[0,1,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,0].set_title('DNS $b_{12}$')
axes[1,0].set(ylabel='$b_{12}$')
fig.colorbar(im, ax = axes[1,0])

im = axes[1,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tau_RANS[0,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,1].set_title('RANS $b_{12}$')
fig.colorbar(im, ax = axes[1,1])

im =axes[1,2].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tauAni[0,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,2].set_title('ML $b_{12}$')
fig.colorbar(im, ax = axes[1,2])


im = axes[2,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], aij_DNS[1,1,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[3,0].set_title('DNS $b_{22}$')
axes[2,0].set(ylabel='$b_{22}$')
fig.colorbar(im, ax = axes[2,0])

im = axes[2,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tau_RANS[1,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[1,1].set_title('RANS $b_{22}$')
fig.colorbar(im, ax = axes[2,1])

im =axes[2,2].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tauAni[1,1,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[3,2].set_title('ML $b_{22}$')
fig.colorbar(im, ax = axes[2,2])

im = axes[3,0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], aij_DNS[2,2,:,:],50, cmap=cmap, extend="both",vmin=-0.5, vmax = 0.5)
#plt.quiver(meshRANS[index1,:,:], meshRANS[index2,:,:], dataRANS_test['U'][1,:,:], dataRANS_test['U'][2,:,:], angles='xy', scale_units='xy', scale=0.2)
#plt.axis('equal')
#axes[5,0].set_title('DNS $b_{33}$')
axes[3,0].set(ylabel='$b_{33}$')
fig.colorbar(im, ax = axes[3,0])


im = axes[3,1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], tau_RANS[2,2,:,:],50,cmap=cmap,extend="both",vmin=-0.5, vmax = 0.5)
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
'''
plt.figure()
newBaryMap = np.add(test_discr_bary, baryMap_RANS) 
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(newBaryMap,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()


plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(baryMap_RANS,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()

print('Difference of the two above plots')
plt.figure()
newBaryMap = np.add(test_discr_bary, baryMap_RANS) 
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], np.add(foam.baryMap_dist(baryMap_RANS,baryMap_DNS), -foam.baryMap_dist(newBaryMap,baryMap_DNS)),100, cmap = "Reds")
plt.colorbar()
plt.show()


plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.phi_dist(phi_discr, test_discr_phi),100, cmap = "Reds")
plt.colorbar()
plt.show()
'''
plt.figure()
newBaryMap = np.add(test_discr_bary, baryMap_RANS) 
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(newBaryMap,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()


plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(baryMap_RANS,baryMap_DNS), 100, cmap = "Reds")
plt.colorbar()
plt.show()

print('Difference of the two above plots')
plt.figure()
newBaryMap = np.add(test_discr_bary, baryMap_RANS) 
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], np.add(foam.baryMap_dist(baryMap_RANS,baryMap_DNS), -foam.baryMap_dist(newBaryMap,baryMap_DNS)),100, cmap = "Reds")
plt.colorbar()
plt.show()


plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.phi_dist(test_discr, test_discr_phi),100, cmap = "Reds")
plt.colorbar()
plt.show()