from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import os
import openFOAM as foam
import sys
sys.path.append("..")

import csv as csv

from tempfile import mkstemp
from shutil import move
from shutil import copy
from os import remove, close
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import export_graphviz


#%%============================================================================
# M A I N   P R O G R A M
#==============================================================================


DO_INTERP   = 1
DO_WRITE    = 0
DO_PLOTTING = 1


# file directories
time_end      = 30000 

#Specify the reynoulds number and turbulence model used (directory)
Re            = 700 #
TurbModel     = 'kOmega'


nx_RANS       = 140
ny_RANS       = 150

#Specify home directory from where the data can be found
home = os.path.realpath('MinorCSE') + '/'
#home = '../CSE minor/'

dir_RANS  = home + ('Re%i_%s' % (Re,TurbModel))


# Load DNS dataset
####################################################################################################

dataset = home + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re) + '/' + ('Hill_Re_%i_Breuer.csv' % Re)
dataDNS = foam.loadData_avg(dataset)



# Load RANS mesh
###################################################################################################
#case_dir      = dir_RANS + case_RANS
meshRANSlist  = foam.getRANSVector(dir_RANS, time_end, 'cellCentres')
meshRANS      = foam.getRANSPlane(meshRANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#velocity
U_RANSlist    = foam.getRANSVector(dir_RANS, time_end, 'U')
U_RANS        = foam.getRANSPlane(U_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#velocity gradient
gradU_RANSlist  = foam.getRANSTensor(dir_RANS, time_end, 'grad(U)')
gradU_RANS      = foam.getRANSPlane(gradU_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')

#pressure
p_RANSlist    = foam.getRANSScalar(dir_RANS, time_end, 'p')
p_RANS        = foam.getRANSPlane(p_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')

#pressure gradient
gradp_RANSlist    = foam.getRANSVector(dir_RANS, time_end, 'grad(p)')
gradp_RANS        = foam.getRANSPlane(gradp_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#Reynolds stress tensor
tau_RANSlist  = foam.getRANSSymmTensor(dir_RANS, time_end, 'R')
tau_RANS      = foam.getRANSPlane(tau_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')

#k
k_RANSlist    = foam.getRANSScalar(dir_RANS, time_end, 'k')
k_RANS        = foam.getRANSPlane(k_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')

#k gradient
gradk_RANSlist    = foam.getRANSVector(dir_RANS, time_end, 'grad(k)')
gradk_RANS        = foam.getRANSPlane(gradk_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')

#distance to wall
yWall_RANSlist = foam.getRANSScalar(dir_RANS, time_end, 'yWall')
yWall_RANS        = foam.getRANSPlane(yWall_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')

#omega
omega_RANSlist  = foam.getRANSScalar(dir_RANS, time_end, 'omega')
omega_RANS      = foam.getRANSPlane(omega_RANSlist, '2D', nx_RANS, ny_RANS, 'scalar')

#S R tensor
S_RANS, Omega_RANS  = foam.getSRTensors(gradU_RANS)


#features
######################################################################################################
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
    #print( "shape urans=", b)
    q2 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):               
            raw = k_RANS[0,i1,i2]
            norm = 0.5*(np.inner(U_RANS[:, i1, i2], U_RANS[:, i1, i2])) # inner is equivalent to sum UiUi
            q2[i1,i2] = raw/(np.abs(raw) + np.abs(norm))
    return q2

    
def q3(k_RANS, yWall_RANS, nu=1.4285714285714286e-03):
    a = np.shape(k_RANS)
    q3 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):               
            q3[i1,i2] = np.minimum((np.sqrt(k_RANS[:,i1,i2][0])*yWall_RANS[:, i1, i2])/(50*nu), 2)
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


def q5(k_RANS, S_RANS, omega_RANS, Cmu=0.09):
    a = np.shape(k_RANS)
    q5 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            epsilon = Cmu * k_RANS[:, i1, i2] * omega_RANS[:, i1, i2]
            raw = k_RANS[:, i1, i2] / epsilon
            norm = 1 / np.sqrt(np.trace(np.dot(S_RANS[:, :, i1, i2],S_RANS[:, :, i1, i2])))
            q5[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q5


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
    

def q7(U_RANS, gradU_RANS):
    a = np.shape(U_RANS)
    q7 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            raw = np.fabs(np.einsum('i, j, ij', U_RANS[:, i1, i2], U_RANS[:, i1, i2], gradU_RANS[:, :, i1, i2]))
            norm = np.sqrt(np.einsum('l, l, i, ij, k, kj', U_RANS[:, i1, i2], U_RANS[:, i1, i2],U_RANS[:, i1, i2], gradU_RANS[:, :, i1, i2], U_RANS[:, i1, i2], gradU_RANS[:, :, i1, i2]))
            q7[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q7



def q8(U, gradK, Tau, S):
    a = np.shape(U)
    q8 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw  = np.einsum('i,i',U[:,i1,i2], gradK[:,i1,i2])
            norm = np.einsum('jk,jk', Tau[:,:, i1, i2], S[:,:, i1, i2])
            q8[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))              
    return q8
  

def q9(tau_RANS, k_RANS):
    a = np.shape(k_RANS)
    q9 = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            raw = np.sqrt(np.trace(np.dot(tau_RANS[:, :, i1, i2],np.transpose(tau_RANS[:, :, i1, i2]))))
            norm = k_RANS[:, i1, i2]
            q9[i1,i2] = raw/(np.fabs(raw) + np.fabs(norm))
    return q9

'''
def getFeatures(Re, TurbModel = 'kOmega', time_end = 30000, nx_RANS = 140, ny_RANS = 150):
    dir = os.path.dirname(__file__)
    home = os.path.realpath('MinorCSE') + '/' #Specify home directory from where the data can be found
    dir_RANS  = home + ('Re%i_%s' % (Re,TurbModel))
    
    a = np.shape(k_RANS)
    feature = np.zeros((9, a[1],a[2]))
    feature[0,:,:] = q1(S_RANS, Omega_RANS)
    feature[1,:,:] = q2(k_RANS, U_RANS)
    feature[2,:,:] = q3(k_RANS, yWall_RANS)
    feature[3,:,:] = q4(U_RANS, gradp_RANS)
    feature[4,:,:] = q5(k_RANS, S_RANS, omega_RANS)
    feature[5,:,:] = q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS)
    feature[6,:,:] = q7(U_RANS, gradU_RANS)
    feature[7,:,:] = q8(U_RANS, gradk_RANS, tau_RANS, S_RANS)
    feature[8,:,:] = q9(tau_RANS, k_RANS)
    
    return feature
  '''  
#this shape is needed for scikit but the other function follows the indices convention in the code.
def getFeatures2(Re, TurbModel = 'kOmega', time_end = 30000, nx_RANS = 140, ny_RANS = 150):
    dir = os.path.dirname(__file__)
    home = os.path.realpath('MinorCSE') + '/' #Specify home directory from where the data can be found
    dir_RANS  = home + ('Re%i_%s' % (Re,TurbModel))
    a = np.shape(k_RANS)
    feature = np.zeros((a[1],a[2], 9))
    feature[:,:,0] = q1(S_RANS, Omega_RANS)
    feature[:,:,1] = q2(k_RANS, U_RANS)
    feature[:,:,2] = q3(k_RANS, yWall_RANS)
    feature[:,:,3] = q4(U_RANS, gradp_RANS)
    feature[:,:,4] = q5(k_RANS, S_RANS, omega_RANS)
    feature[:,:,5] = q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS)
    feature[:,:,6] = q7(U_RANS, gradU_RANS)
    feature[:,:,7] = q8(U_RANS, gradk_RANS, tau_RANS, S_RANS)
    feature[:,:,8] = q9(tau_RANS, k_RANS)
    return feature
 


# interpolate DNS on RANS grid
if DO_INTERP:
    dataDNS_i = foam.interpDNSOnRANS(dataDNS, meshRANS)

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

for i in range(l1):
    for j in range(l2):
        aij_DNS[:,:,i,j] = ReStress_DNS[:,:,i,j]/(2.*dataDNS_i['k'][i,j]) - np.diag([1/3.,1/3.,1/3.])
        dataRANS_k[i,j] = 0.5 * np.trace(tau_RANS[:,:,i,j])
        dataRANS_aij[:,:,i,j] = tau_RANS[:,:,i,j]/(2.*dataRANS_k[i,j]) - np.diag([1/3.,1/3.,1/3.])
        



eigVal_DNS = foam.calcEigenvalues(ReStress_DNS, dataDNS_i['k'])
baryMap_DNS = foam.barycentricMap(eigVal_DNS)

eigVal_RANS = foam.calcEigenvalues(tau_RANS, dataRANS_k)
baryMap_RANS = foam.barycentricMap(eigVal_RANS)

baryMap_discr= foam.baryMap_discr(baryMap_RANS, baryMap_DNS)



#%% write out OpenFOAM data files with DNS interpolated on RANS
if DO_WRITE:
# pressure p

    case = dir_RANS + 'case_simpleFoam'
    time = 0
    var = 'pd'
    data = np.swapaxes(dataDNS_i['pm'],0,1).reshape(nx_RANS*ny_RANS)
    
    copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var + '_old')
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    cc = False
    j = 0
    file_path=case + '/' + str(time) + '/' + var
    print (file_path)
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as file:
            for i,line in enumerate(file):  
                if 'object' in line:
                    new_file.write('    object      p;\n')
                elif cc==False and 'internalField' not in line:
                    new_file.write(line)
                
                elif 'internalField' in line:
                    tmp = i + 1
                    tmp2 = i + 3
                    cc = True
                    new_file.write(line)
                    print (tmp, tmp2)
                    
        
                elif i==tmp:
                    print (line.split())
                    maxLines = int(line.split()[0])
                    maxIter  = tmp2 + maxLines
                    new_file.write(line)
                    print (maxLines, maxIter)
                
                elif i>tmp and i<tmp2:              
                    new_file.write(line)
                
                elif i>=tmp2 and i<maxIter:
                    new_file.write( str(data[j]) + ' \n'  )            
                    j += 1
                
                elif i>=maxIter:
                    new_file.write(line)
                    
    close(fh)
    remove(file_path)
    move(abs_path, file_path)


# velocity
    time = 0
    var = 'Ud'

    data_x = np.swapaxes(dataDNS_i['um'],0,1).reshape(nx_RANS*ny_RANS)
    data_y = np.swapaxes(dataDNS_i['vm'],0,1).reshape(nx_RANS*ny_RANS)
    data_z = np.swapaxes(dataDNS_i['wm'],0,1).reshape(nx_RANS*ny_RANS)
    

    copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var + '_old')
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    cc = False
    j = 0
    file_path=case + '/' + str(time) + '/' + var
    print (file_path)
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as file:
            for i,line in enumerate(file):  
                if 'object' in line:
                    new_file.write('    object      U;\n')
                elif cc==False and 'internalField' not in line:
                    new_file.write(line)
                
                elif 'internalField' in line:
                    tmp = i + 1
                    tmp2 = i + 3
                    cc = True
                    new_file.write(line)
                    print(tmp, tmp2)
                    
        
                elif i==tmp:
                    print (line.split())
                    maxLines = int(line.split()[0])
                    maxIter  = tmp2 + maxLines
                    new_file.write(line)
                    print (maxLines, maxIter)
                
                elif i>tmp and i<tmp2:              
                    new_file.write(line)
                
                elif i>=tmp2 and i<maxIter:
                    #print line
                    new_file.write('(' + str(data_x[j]) + ' ' +  str(data_y[j]) + ' ' + str(data_z[j]) + ') \n'  )
                    j += 1
                
                elif i>=maxIter:
                    new_file.write(line)
                    
    close(fh)
    remove(file_path)
    move(abs_path, file_path)


#%% PLOTTING
if DO_PLOTTING:
    
    plt.close('all')
    
    plt.figure()
    plt.title("Velocity from DNS")
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['um'],20)
    plt.show()
    
    plt.figure()
    plt.title("Velocity from RANS")
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], U_RANS[0,:,:],20)
    plt.show()
    
    plt.figure()
    plt.title("k from DNS")
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['k'],20)
    plt.show()
    
    plt.figure()
    plt.title("k from RANS")
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS_k,20)
    plt.show()
    
    plt.figure()
    plt.title("DNS")
    plt.plot(baryMap_DNS[0,:,:],baryMap_DNS[1,:,:],'b*')
    plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
    plt.axis('equal')
    plt.show()
    
    plt.figure()
    plt.title("RANS")
    plt.plot(baryMap_RANS[0,:,:],baryMap_RANS[1,:,:],'b*')
    plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
    plt.axis('equal')
    plt.show()
   
    plt.figure()
    plt.title("Discripancy")
    plt.plot(baryMap_discr[0,:,:],baryMap_discr[1,:,:],'b*')
    plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
    plt.axis('equal')
    plt.show()
    
    plt.figure()
    plt.title("dist")
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(baryMap_RANS,baryMap_DNS))
    plt.show()
    
    plt.figure()
    plt.title("Feature 6")
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS))
    plt.show()
'''
aaa = getFeatures2()
print(np.shape(aaa))
print(aaa)
bbb = foam.baryMap_discr(baryMap_RANS, baryMap_DNS)
print(np.shape(bbb))
print(bbb)
'''
nx = 140
ny = 150
f = getFeatures2(700)
X = np.reshape(f, (nx*ny, 9))

y = np.reshape(foam.baryMap_dist(baryMap_RANS, baryMap_DNS), (nx*ny, 1))


regr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

regr.fit(X, y)

print("Feature importance :", regr.feature_importances_)   

print("Feature at x=33, y=12:",f[45, 12, :])

print("Prediction for distance:", regr.predict([f[45, 12, :]]))

print("Real distance", foam.baryMap_dist(baryMap_RANS, baryMap_DNS)[45][12])


nx = 2
ny = 3
a = np.zeros((4, nx, ny))
q1 = np.array([[1, 1, 1],[2, 2, 2]])

q2 = np.array([[3, 3, 3],[4, 4, 4]])
q3 = np.array([[5, 5 , 5],[6, 6, 6]])
q4 = np.array([[7, 7, 7],[8, 8, 8]])
a[0,:, :] = q1
a[1, :, :] = q2
a[2,:, :] = q3
a[3, :, :] = q4

#q = np.reshape(a.swapaxes(1,2), (6, 4), "F")
#print(q)
#b= np.reshape(q.swapaxes(1,0), (6, 4))
#print(b)

b = a*2
print(b)
c = np.zeros((2, 4, nx, ny))
c[0, :, :, :] = a
c[1, :, :, :] = b
print(c)
c = np.reshape(np.reshape(c.swapaxes(1, 2), (12, 4), "F").swapaxes(1,0), (12,4))
print(c)





