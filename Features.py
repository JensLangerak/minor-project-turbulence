from __future__ import division
import numpy as np
import os
import openFOAM as foam
    
# Load RANS mesh
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



nu=1.4285714285714286e-03

def q3(k_RANS, yWall_RANS, nu):
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


#print(q9(tau_RANS, k_RANS))

'''
plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], q4(U_RANS, gradp_RANS))
plt.show()

plt.figure()
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS))
plt.show()
'''

def getFeatures(Re, TurbModel = 'kOmega', time_end = 30000, nx_RANS = 140, ny_RANS = 150):
    dir = os.path.dirname(__file__)
    home = os.path.realpath('MinorCSE') + '/' #Specify home directory from where the data can be found
    dir_RANS  = home + ('Re%i_%s' % (Re,TurbModel))
    
    a = np.shape(k_RANS)
    feature = np.zeros((9, a[1],a[2]))
    feature[0,:,:] = q1(S_RANS, Omega_RANS)
    feature[1,:,:] = q2(k_RANS, U_RANS)
    feature[2,:,:] = q3(k_RANS, yWall_RANS, nu)
    feature[3,:,:] = q4(U_RANS, gradp_RANS)
    feature[4,:,:] = q5(k_RANS, S_RANS, Cmu, omega_RANS)
    feature[5,:,:] = q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS)
    feature[6,:,:] = q7(U_RANS, gradU_RANS)
    feature[7,:,:] = q8(U_RANS, gradk_RANS, tau_RANS, S_RANS)
    feature[8,:,:] = q9(tau_RANS, k_RANS)
    
    return feature
    

#def getFeatures2(): #this shape is needed for scikit but the other function follows the indices convention in the code.
#    a = np.shape(k_RANS)
#    feature = np.zeros((9, a[1],a[2]))
#    feature[:,:,0] = q1(S_RANS, Omega_RANS)
#    feature[:,:,1] = q2(k_RANS, U_RANS)
#    feature[:,:,2] = q3(k_RANS, yWall_RANS, nu)
#    feature[:,:,3] = q4(U_RANS, gradp_RANS)
#    feature[:,:,4] = q5(k_RANS, S_RANS, Cmu, omega_RANS)
#    feature[:,:,5] = q6(gradp_RANS, gradU_RANS, p_RANS,U_RANS)
#    feature[:,:,6] = q7(U_RANS, gradU_RANS)
#    feature[:,:,7] = q8(U_RANS, gradk_RANS, tau_RANS, S_RANS)
#    feature[:,:,8] = q9(tau_RANS, k_RANS)
#    return feature
    
test = getFeatures(1400)
t2 = getFeatures(1400)
t2 = np.reshape(t2,(9, 140 * 150))



