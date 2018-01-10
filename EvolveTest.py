from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import openFOAM as foam
import os
import sys
import evolution as ev
sys.path.append("..")
from sklearn.ensemble import RandomForestRegressor

import sys

sys.path.append("..")

home = os.path.realpath('MinorCSE') + '/'


##################################################################################################################
######################################### Loading the RANS data ##################################################
##################################################################################################################
def RANS(case, Re, TurbModel, time_end, nx, ny):
    dir_RANS = home + ('%s' % case) + '/' + ('Re%i_%s' % (Re, TurbModel))
    if case == 'SquareDuct':
        dir_RANS = dir_RANS + '_50'

    mesh_list = foam.getRANSVector(dir_RANS, time_end, 'cellCentres')
    mesh = foam.getRANSPlane(mesh_list, '2D', nx, ny, 'vector')
    # velocity
    U_list = foam.getRANSVector(dir_RANS, time_end, 'U')
    U = foam.getRANSPlane(U_list, '2D', nx, ny, 'vector')
    # velocity gradient
    gradU_list = foam.getRANSTensor(dir_RANS, time_end, 'grad(U)')
    gradU = foam.getRANSPlane(gradU_list, '2D', nx, ny, 'tensor')
    # pressure
    p_list = foam.getRANSScalar(dir_RANS, time_end, 'p')
    p = foam.getRANSPlane(p_list, '2D', nx, ny, 'scalar')
    # pressure gradient
    gradp_list = foam.getRANSVector(dir_RANS, time_end, 'grad(p)')
    gradp = foam.getRANSPlane(gradp_list, '2D', nx, ny, 'vector')
    # Reynolds stress tensor
    tau_list = foam.getRANSSymmTensor(dir_RANS, time_end, 'R')
    tau = foam.getRANSPlane(tau_list, '2D', nx, ny, 'tensor')
    # k
    k_list = foam.getRANSScalar(dir_RANS, time_end, 'k')
    k = foam.getRANSPlane(k_list, '2D', nx, ny, 'scalar')
    # k gradient
    gradk_list = foam.getRANSVector(dir_RANS, time_end, 'grad(k)')
    gradk = foam.getRANSPlane(gradk_list, '2D', nx, ny, 'vector')
    # distance to wall
    yWall_list = foam.getRANSScalar(dir_RANS, time_end, 'yWall')
    yWall = foam.getRANSPlane(yWall_list, '2D', nx, ny, 'scalar')
    # omega
    omega_list = foam.getRANSScalar(dir_RANS, time_end, 'omega')
    omega = foam.getRANSPlane(omega_list, '2D', nx, ny, 'scalar')
    # S R tensor
    S, Omega = foam.getSRTensors(gradU)

    return mesh, U, gradU, p, gradp, tau, k, gradk, yWall, omega, S, Omega


##################################################################################################################
######################################### Features  ##############################################################
##################################################################################################################
def q1(S, Omega):
    a = np.shape(S)
    q1 = np.zeros((a[2], a[3]))
    for i1 in range(a[2]):
        for i2 in range(a[3]):
            raw = 0.5 * (np.abs(np.trace(np.dot(S[:, :, i1, i2], S[:, :, i1, i2]))) - np.abs(
                np.trace(np.dot(Omega[:, :, i1, i2], -1 * (Omega[:, :, i1, i2])))))
            norm = np.trace(np.dot(S[:, :, i1, i2], S[:, :, i1, i2]))
            q1[i1, i2] = raw / (np.abs(raw) + np.abs(norm))
    return q1


def q2(k, U):
    a = np.shape(k)
    q2 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw = k[0, i1, i2]
            norm = 0.5 * (np.inner(U[:, i1, i2], U[:, i1, i2]))  # inner is equivalent to sum UiUi
            q2[i1, i2] = raw / (np.abs(raw) + np.abs(norm))
    return q2


def q3(k, yWall, nu=1.4285714285714286e-03):
    a = np.shape(k)
    q3 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            q3[i1, i2] = np.minimum((np.sqrt(k[:, i1, i2][0]) * yWall[:, i1, i2]) / (50 * nu), 2)
    return q3


def q4(U, gradP):
    a = np.shape(gradP)
    q4 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw = np.einsum('k,k', U[:, i1, i2], gradP[:, i1, i2])
            norm = np.einsum('j,j,i,i', gradP[:, i1, i2], gradP[:, i1, i2], U[:, i1, i2], U[:, i1, i2])

            q4[i1, i2] = raw / (np.fabs(norm) + np.fabs(raw));
    return q4


def q5(k, S, omega, Cmu=0.09):
    a = np.shape(k)
    q5 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            epsilon = Cmu * k[:, i1, i2] * omega[:, i1, i2]
            raw = k[:, i1, i2] / epsilon
            norm = 1 / np.sqrt(np.trace(np.dot(S[:, :, i1, i2], S[:, :, i1, i2])))
            q5[i1, i2] = raw / (np.fabs(raw) + np.fabs(norm))
    return q5


def q6(gradP, gradU, p, U):
    a = np.shape(gradP)
    q6 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw = np.sqrt(np.einsum('i,i', gradP[:, i1, i2], gradP[:, i1, i2]))
            norm = np.einsum('k, kk', U[:, i1, i2], gradU[:, :, i1, i2])

            norm *= 0.5 * p[0, i1, i2]
            q6[i1, i2] = raw / (np.fabs(raw) + np.fabs(norm))
    return q6


def q7(U, gradU):
    a = np.shape(U)
    q7 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw = np.fabs(np.einsum('i, j, ij', U[:, i1, i2], U[:, i1, i2], gradU[:, :, i1, i2]))
            norm = np.sqrt(
                np.einsum('l, l, i, ij, k, kj', U[:, i1, i2], U[:, i1, i2], U[:, i1, i2], gradU[:, :, i1, i2],
                          U[:, i1, i2], gradU[:, :, i1, i2]))
            q7[i1, i2] = raw / (np.fabs(raw) + np.fabs(norm))
    return q7


def q8(U, gradK, Tau, S):
    a = np.shape(U)
    q8 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw = np.einsum('i,i', U[:, i1, i2], gradK[:, i1, i2])
            norm = np.einsum('jk,jk', Tau[:, :, i1, i2], S[:, :, i1, i2])
            q8[i1, i2] = raw / (np.fabs(raw) + np.fabs(norm))
    return q8


def q9(tau, k):
    a = np.shape(k)
    q9 = np.zeros((a[1], a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):
            raw = np.sqrt(np.trace(np.dot(tau[:, :, i1, i2], np.transpose(tau[:, :, i1, i2]))))
            norm = k[:, i1, i2]
            q9[i1, i2] = raw / (np.fabs(raw) + np.fabs(norm))
    return q9


##################################################################################################################
######################################### Feature function #######################################################
##################################################################################################################

def features(case, Re, TurbModel, time_end, nx, ny):
    X = np.zeros((nx * len(Re) * ny, 9))

    for i in range(len(Re)):
        meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(
            case, Re[i], TurbModel, time_end, nx, ny)
        feature = np.zeros((9, nx, ny))
        feature[0, :, :] = q1(S_RANS, Omega_RANS)
        feature[1, :, :] = q2(k_RANS, U_RANS)
        feature[2, :, :] = q3(k_RANS, yWall_RANS)
        feature[3, :, :] = q4(U_RANS, gradp_RANS)
        feature[4, :, :] = q5(k_RANS, S_RANS, omega_RANS)
        feature[5, :, :] = q6(gradp_RANS, gradU_RANS, p_RANS, U_RANS)
        feature[6, :, :] = q7(U_RANS, gradU_RANS)
        feature[7, :, :] = q8(U_RANS, gradk_RANS, tau_RANS, S_RANS)
        feature[8, :, :] = q9(tau_RANS, k_RANS)
        feature = np.reshape(feature.swapaxes(1, 2), (nx * ny, 9), "F")
        feature = np.reshape(feature.swapaxes(1, 0), (nx * ny, 9))
        X[i * nx * ny:(i + 1) * nx * ny, :] = feature
    return X


##################################################################################################################
##################################### Eigenvalue discripancy function ############################################
##################################################################################################################

def response(case, Re, TurbModel, time_end, nx, ny, train):
    if train == True:
        print('train = true')
        Y = np.zeros((nx * len(Re) * ny, 2))
        for i in range(len(Re)):
            dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re[i]) + '/' + (
                    'Hill_Re_%i_Breuer.csv' % Re[i])
            meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(
                case, Re[i], TurbModel, time_end, nx, ny)

            dataDNS = foam.loadData_avg(dataset)
            dataDNS_i = foam.interpDNSOnRANS(dataDNS, meshRANS)
            dataDNS_i['k'] = 0.5 * (dataDNS_i['uu'] + dataDNS_i['vv'] + dataDNS_i['ww'])

            l1 = np.shape(U_RANS)[1]
            l2 = np.shape(U_RANS)[2]

            ReStress_DNS = np.zeros([3, 3, l1, l2])
            ReStress_DNS[0, 0, :, :] = dataDNS_i['uu']
            ReStress_DNS[1, 1, :, :] = dataDNS_i['vv']
            ReStress_DNS[2, 2, :, :] = dataDNS_i['ww']
            ReStress_DNS[0, 1, :, :] = dataDNS_i['uv']
            ReStress_DNS[1, 0, :, :] = dataDNS_i['uv']

            aij_DNS = np.zeros([3, 3, l1, l2])
            dataRANS_k = np.zeros([l1, l2])
            dataRANS_aij = np.zeros([3, 3, l1, l2])

            for j in range(l1):
                for k in range(l2):
                    aij_DNS[:, :, j, k] = ReStress_DNS[:, :, j, k] / (2. * dataDNS_i['k'][j, k]) - np.diag(
                        [1 / 3., 1 / 3., 1 / 3.])
                    dataRANS_k[j, k] = 0.5 * np.trace(tau_RANS[:, :, j, k])
                    dataRANS_aij[:, :, j, k] = tau_RANS[:, :, j, k] / (2. * dataRANS_k[j, k]) - np.diag(
                        [1 / 3., 1 / 3., 1 / 3.])

            aneigVal_DNS = foam.calcEigenvalues(aij_DNS, dataDNS_i['k'])
            baryMap_DNS = foam.barycentricMap(aneigVal_DNS)

            aneigVal_RANS = foam.calcEigenvalues(dataRANS_aij, dataRANS_k)
            baryMap_RANS = foam.barycentricMap(aneigVal_RANS)

            eigVal_RANS, eigVec_RANS = foam.eigenDecomposition(dataRANS_aij)
            eigVal_DNS, eigVec_DNS = foam.eigenDecomposition(aij_DNS)

            phi_RANS = np.reshape(
                (np.reshape((foam.eigenvectorToEuler(eigVec_RANS)).swapaxes(1, 2), (nx * ny, 3), "F")).swapaxes(1, 0),
                (nx * ny, 3))
            phi_DNS = np.reshape(
                (np.reshape((foam.eigenvectorToEuler(eigVec_DNS)).swapaxes(1, 2), (nx * ny, 3), "F")).swapaxes(1, 0),
                (nx * ny, 3))

            baryMap_discr = np.reshape((np.reshape((foam.baryMap_discr(baryMap_RANS, baryMap_DNS)).swapaxes(1, 2),
                                                   (nx * ny, 2), "F")).swapaxes(1, 0), (nx * ny, 2))

            phi_discr = phi_DNS - phi_RANS
            k_discr = np.reshape(
                (np.reshape((dataDNS_i['k'] - k_RANS).swapaxes(1, 2), (nx * ny, 1), "F")).swapaxes(1, 0), (nx * ny, 1))

            Y[i * nx * ny:(i + 1) * nx * ny, 0:2] = baryMap_discr
           # Y[i * nx * ny:(i + 1) * nx * ny, 2:5] = phi_discr
           # Y[i * nx * ny:(i + 1) * nx * ny, 5] = k_discr[:, 0]

        print('return Y')
        return Y

    else:
        print('train = false')
        for i in range(len(Re)):
            dataset = home + ('%s' % (case)) + '/' + ('DATA_CASE_LES_BREUER') + '/' + ('Re_%i' % Re[i]) + '/' + (
                    'Hill_Re_%i_Breuer.csv' % Re[i])
            meshRANS, U_RANS, gradU_RANS, p_RANS, gradp_RANS, tau_RANS, k_RANS, gradk_RANS, yWall_RANS, omega_RANS, S_RANS, Omega_RANS = RANS(
                case, Re[i], TurbModel, time_end, nx, ny)

            dataDNS = foam.loadData_avg(dataset)
            dataDNS_i = foam.interpDNSOnRANS(dataDNS, meshRANS)
            dataDNS_i['k'] = 0.5 * (dataDNS_i['uu'] + dataDNS_i['vv'] + dataDNS_i['ww'])

            l1 = np.shape(U_RANS)[1]
            l2 = np.shape(U_RANS)[2]

            ReStress_DNS = np.zeros([3, 3, l1, l2])
            ReStress_DNS[0, 0, :, :] = dataDNS_i['uu']
            ReStress_DNS[1, 1, :, :] = dataDNS_i['vv']
            ReStress_DNS[2, 2, :, :] = dataDNS_i['ww']
            ReStress_DNS[0, 1, :, :] = dataDNS_i['uv']
            ReStress_DNS[1, 0, :, :] = dataDNS_i['uv']

            aij_DNS = np.zeros([3, 3, l1, l2])
            dataRANS_k = np.zeros([l1, l2])
            dataRANS_aij = np.zeros([3, 3, l1, l2])

            for j in range(l1):
                for k in range(l2):
                    aij_DNS[:, :, j, k] = ReStress_DNS[:, :, j, k] / (2. * dataDNS_i['k'][j, k]) - np.diag(
                        [1 / 3., 1 / 3., 1 / 3.])
                    dataRANS_k[j, k] = 0.5 * np.trace(tau_RANS[:, :, j, k])
                    dataRANS_aij[:, :, j, k] = tau_RANS[:, :, j, k] / (2. * dataRANS_k[j, k]) - np.diag(
                        [1 / 3., 1 / 3., 1 / 3.])

            eigVal_DNS = foam.calcEigenvalues(aij_DNS, dataDNS_i['k'])
            print(np.shape(eigVal_DNS))
            baryMap_DNS = foam.barycentricMap(eigVal_DNS)
            print(np.shape(baryMap_DNS))

            eigVal_RANS = foam.calcEigenvalues(dataRANS_aij, dataRANS_k)
            baryMap_RANS = foam.barycentricMap(eigVal_RANS)
            print(np.shape(baryMap_RANS))
            baryMap_discr = foam.baryMap_discr(baryMap_RANS, baryMap_DNS)
            print(np.shape(baryMap_discr))
            print('return bary')
            return baryMap_RANS, baryMap_DNS, baryMap_discr


##################################################################################################################
##################################################################################################################
######################################### Random forest ##########################################################
##################################################################################################################

case = 'PeriodicHills'

# Training
Re = [700, 1400, 2800, 5600, 10595]
Re_train = np.delete(Re, 0)
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

'''case = 'ConvergingDivergingChannel'
Re = [12600]
TurbModel = ='kOmega'
dir_RANS  = home + ('%s' % case) + '/' + ('Re%i_%s_100' % (Re,TurbModel))
f = features(Re, TurbModel='kOmega', time_end=7000, nx=140, ny=100)
'''

Y_train = response('PeriodicHills', Re_train, TurbModel='kOmega', time_end=30000, nx=140, ny=150, train=True)

#regr = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,
#                             min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
#                             min_impurity_decrease=0.0,
#                             min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
#                             verbose=0, warm_start=False)

#regr.fit(X_train, Y_train)
#print("Feature importance :", regr.feature_importances_)

evolver = ev.GCPEvolver(half_population=5, nr_nodes=15, mutation_chance=0.02, max_score=700000000, nr_features=9)

sol = evolver.evolve(X_train, Y_train)


# Testing
Re_test = [Re[0]]
print(Re_test)

test_X = features('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000, nx=140, ny=150)
test_discr = evolver.predict(sol, test_X)
test_discr = np.reshape(test_discr.swapaxes(1, 0), (2, 140, 150))

baryMap_RANS, baryMap_DNS, baryMap_discr = response('PeriodicHills', Re_test, TurbModel='kOmega', time_end=30000,
                                                    nx=140, ny=150, train=False)

# Plots
tra = ev.cgp.complete_translate(sol, 9, 10)
print (tra[10 + 9 -1])
print (tra[10 + 9 -2])

plt.figure()
plt.title('DNS %s_Re%i' % (case, Re_test[0]))
plt.plot(baryMap_DNS[0, :, :], baryMap_DNS[1, :, :], 'b*')
plt.plot([0, 1, 0.5, 0], [0, 0, np.sin(60 * (np.pi / 180)), 0], 'k-')
plt.axis('equal')
plt.show()

plt.figure()
plt.title("RANS %s_Re%i corrected with predicted descripancies from RF" % (case, Re_test[0]))
plt.plot(np.add(test_discr[0, :, :], baryMap_RANS[0, :, :]), np.add(test_discr[1, :, :], baryMap_RANS[1, :, :]), 'b*')
plt.plot([0, 1, 0.5, 0], [0, 0, np.sin(60 * (np.pi / 180)), 0], 'k-')
plt.axis('equal')
plt.show()

plt.figure()
plt.title("RANS %s_Re%i" % (case, Re_test[0]))
plt.plot(baryMap_RANS[0, :, :], baryMap_RANS[1, :, :], 'b*')
plt.plot([0, 1, 0.5, 0], [0, 0, np.sin(60 * (np.pi / 180)), 0], 'k-')
plt.axis('equal')
plt.show()

plt.figure()
plt.title("Eigenvalue discripancy %s_Re%i" % (case, Re_test[0]))
plt.plot(baryMap_discr[0, :, :], baryMap_discr[1, :, :], 'b*')
plt.plot([0, 1, 0.5, 0], [0, 0, np.sin(60 * (np.pi / 180)), 0], 'k-')
plt.axis('equal')
plt.show()

plt.figure()
plt.title("Discripancy after RF %s_Re%i" % (case, Re_test[0]))
plt.plot(test_discr[0, :, :], test_discr[1, :, :], 'b*')
plt.plot([0, 1, 0.5, 0], [0, 0, np.sin(60 * (np.pi / 180)), 0], 'k-')
plt.axis('equal')
plt.show()

input()

'''
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
plt.title("dist")
plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], foam.baryMap_dist(baryMap_RANS,baryMap_DNS))
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
            test_discr = np.reshape(test_discr.swapaxes(1,0), (2, 140, 150))
            Y_test = response('PeriodicHills', Re_test, nx=140, ny=150, train = True)
            score[i, j] = regr.score(test_X, Y_test)
    return score

Nest = [3, 6, 9, 12]
Nfeatures = [3, 5, 7, 9]

p = plotRF(Nest, Nfeatures, X_train, Y_train, [700])

plt.figure()
plt.title("score, number of estimators")
plt.plot(Nest, p[:,0])
plt.show()

plt.figure()
plt.title("score, number of features")
plt.plot(Nfeatures, p[0,:])
plt.show()

plt.figure()
plt.title("score")
plt.contourf(Nfeatures, Nest, p)
plt.ylabel("Number of estimators")
plt.xlabel("Number of features")
plt.colorbar()
plt.show()

'''