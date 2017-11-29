 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:04:13 2017

@author: mikael
"""

#===============================================================================
# S U B R O U T I N E S
#===============================================================================
def loadData_avg(dataset):
    data_list = np.zeros([10, 100000])
    with open(dataset, 'r') as f:
        reader = csv.reader(f)
        names = next(reader)[:-1]
        for i,row in enumerate(reader):
            if row:
                data_list[:,i] = np.array([float(ii) for ii in row[:-1]])
    
    data_list = data_list[:,:i]
    
    data = {}
    for i,var in enumerate(names):
        data[var] = data_list[i,:]
                
    return data


def interpDNSOnRANS(dataDNS, meshRANS):
    names = dataDNS.keys()    
    data = {}
    xy = np.array([dataDNS['x'], dataDNS['y']]).T
    for var in names:
        if not var=='x' and not var=='y':        
            data[var] = interp.griddata(xy, dataDNS[var], (meshRANS[0,:,:], meshRANS[1,:,:]), method='linear')

    return data
    
def writeP(case, time, var, data):
    #file = open(case + '/' + str(time) + '/' + var,'r').readlines()
    copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var)
    #file = open('R~','r+').readlines()
    #file = open('wavyWall_Re6850_komegaSST_4L_2D/20000/gradU','r').readlines()
    #lines=0
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    #v= np.zeros([3,70000])
    cc = False
    j = 0
    file_path=case + '/' + str(time) + '/' + var
    print( file_path)
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
                    print( tmp, tmp2)
                    
        
                elif i==tmp:
                    print (line.split())
                    maxLines = int(line.split())
                    maxIter  = tmp2 + maxLines
                    #v = np.zeros([3,3,maxLines])
                    new_file.write(line)
                    print (maxLines, maxIter)
                
                elif i>tmp and i<tmp2:              
                    new_file.write(line)
                
                elif i>=tmp2 and i<maxIter:
                    #print line
                    new_file.write( str(data[0,0,j]) + '\n'  )
                    j += 1
                
                elif i>=maxIter:
                    new_file.write(line)
                    
    close(fh)
    remove(file_path)
    move(abs_path, file_path)


def writeReynoldsStressField(ReStress,home,Re,turbModel,nx_RANS,ny_RANS,time_end,suffix):
    # write the calculated Reynolds stress field (y_predict) to OpenFOAM format file
    # prerequisites: a file named 'R' or 'turbulenceProperties:R' in the time_end directory, which will be used 
    # as a template for the adjusted reynolds stress
    case  = home + ('Re%i_%s_%i' % (Re,turbModel,nx_RANS))
    time = time_end
    var = 'R' + suffix
    

    tau_0 = np.swapaxes(ReStress[0,0,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_1 = np.swapaxes(ReStress[0,1,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_2 = np.swapaxes(ReStress[0,2,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_3 = np.swapaxes(ReStress[1,1,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_4 = np.swapaxes(ReStress[1,2,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_5 = np.swapaxes(ReStress[2,2,:,:],0,1).reshape(nx_RANS*ny_RANS)
    
    # if Ud exists, back it up, otherwise use U as template for Ud
    if os.path.exists(case + '/' + str(time) + '/' + var) == True:
        copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var + '_old')
    else:
        copy(case + '/' + str(time) + '/' + 'turbulenceProperties:R', case + '/' + str(time) + '/' + var)
    
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
                    new_file.write('    object      R;\n')
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
                    new_file.write('(' + str(tau_0[j]) + ' ' +  str(tau_1[j]) + ' ' + str(tau_2[j]) + ' ' +  str(tau_3[j]) + ' ' +  str(tau_4[j]) + ' ' +  str(tau_5[j]) +') \n'  )
                    j += 1
                
                elif i>=maxIter:
                    new_file.write(line)
                    
    close(fh)
    remove(file_path)
    move(abs_path, file_path)


#%%============================================================================
# M A I N   P R O G R A M
#==============================================================================
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

DO_INTERP   = 1
DO_WRITE    = 0
DO_PLOTTING = 1


# file directories
nx_RANS       = 140
ny_RANS       = 150
time_end      = 30000
Re            = 700
TurbModel     = 'kOmega'
#home = '/home/mikael/stack/AeroMSc/Thesis/Python/PeriodicHills/'
home = '../'

dir_DNS   = home + ('DATA_CASE_LES_BREUER/Re_%i/' % Re)
dir_RANS  = home + ('Re%i_%s' % (Re,TurbModel))
#case_RANS = 'case_simpleFoam/'

# Load DNS dataset
dataset = dir_DNS + ('Hill_Re_%i_Breuer.csv' % Re)
dataDNS = loadData_avg(dataset)

# Load RANS mesh
#case_dir      = dir_RANS + case_RANS
meshRANSlist  = pyfoam.getRANSVector(dir_RANS, time_end, 'cellCentres')
meshRANS      = pyfoam.getRANSPlane(meshRANSlist,'2D', nx_RANS, ny_RANS, 'vector')
U_RANSlist    = pyfoam.getRANSVector(dir_RANS, time_end, 'U')
U_RANS        = pyfoam.getRANSPlane(U_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')
p_RANSlist    = pyfoam.getRANSScalar(dir_RANS, time_end, 'p')
p_RANS        = pyfoam.getRANSPlane(p_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')
tau_RANSlist  = pyfoam.getRANSSymmTensor(dir_RANS, time_end, 'R')
tau_RANS      = pyfoam.getRANSPlane(tau_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')

# interpolate DNS on RANS grid
if DO_INTERP:
    dataDNS_i = interpDNSOnRANS(dataDNS, meshRANS)

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
        
eigVal_DNS = pyfoam.calcEigenvalues(ReStress_DNS, dataDNS_i['k'])
baryMap_DNS = pyfoam.barycentricMap(eigVal_DNS)

eigVal_RANS = pyfoam.calcEigenvalues(tau_RANS, dataRANS_k)
baryMap_RANS = pyfoam.barycentricMap(eigVal_RANS)

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
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['um'],20)
    plt.show()
    
    plt.figure()
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], U_RANS[0,:,:],20)
    plt.show()
    
    
    plt.figure()
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['k'],20)
    plt.show()
    
    plt.figure()
    plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS_k,20)
    plt.show()
    
    plt.figure()
    plt.plot(baryMap_DNS[0,:,:],baryMap_DNS[1,:,:],'b*')
    plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
    plt.axis('equal')
    plt.show()
    
    plt.figure()
    plt.plot(baryMap_RANS[0,:,:],baryMap_RANS[1,:,:],'b*')
    plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
    plt.axis('equal')
    plt.show()
    

            