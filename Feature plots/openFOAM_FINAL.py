from __future__ import division
import numpy as np
import os
import csv as csv
import scipy.interpolate as interp
from transforms3d.euler import euler2mat, mat2euler

#===============================================================================
# S U B R O U T I N E S
#===============================================================================



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
            #print (tmp, tmp2)
        elif i==tmp:
            maxLines = int(line.split()[0])
            maxIter  = tmp2 + maxLines
            v = np.zeros([1,maxLines])
            #print (maxLines, maxIter)
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
            #print (tmp, tmp2)
        elif i==tmp:
            maxLines = int(line.split()[0])
            maxIter  = tmp2 + maxLines
            v = np.zeros([3,3,maxLines])
            #print (maxLines, maxIter)
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
            #print (tmp, tmp2)
        elif i==tmp:
            maxLines = int(line.split()[0])
            maxIter  = tmp2 + maxLines
            v = np.zeros([3,3,maxLines])
            #print( maxLines, maxIter)
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
    #print( mesh.shape)

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
    #print (mesh.shape)

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
    #print(a);
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

def calcEigenvalues(tau, k):

    if len(tau.shape)==3:
        l=tau.shape[2]

        tauAni = np.zeros([3,3,l])
        for i in range(l):
                tauAni[:,:,i] = tau[:,:,i]/(2.*k[i]) - np.diag([1/3.,1/3.,1/3.])

        eigVal = np.zeros([3,l])
        for i in range(l):
            a,b=np.linalg.eig(tauAni[:,:,i])
            eigVal[:,i]=sorted(a, reverse=True)


    elif len(tau.shape)==4:
        l=tau.shape[2]
        l2=tau.shape[3]
        print (l,l2)
        tauAni = np.zeros([3,3,l,l2])
        for i in range(l):
            for j in range(l2):
                tauAni[:,:,i,j] = tau[:,:,i,j]/(2.*k[i,j]) - np.diag([1/3.,1/3.,1/3.])

        eigVal = np.zeros([3,l,l2])
        for i in range(l):
            for j in range(l2):
                a,b=np.linalg.eig(tauAni[:,:,i,j])
                eigVal[:,i,j]=sorted(a, reverse=True)


    return eigVal


# returns x any y coordinates on barycentric map from the eigenvalues of anisotropy tensor
def barycentricMap(eigVal):

    if len(eigVal.shape)==2:
        l=eigVal.shape[1]

        C1c = eigVal[0,:] - eigVal[1,:]
        C2c = 2*(eigVal[1,:] - eigVal[2,:])
        C3c = 3*eigVal[2,:] + 1
        Cc  = np.array([C1c, C2c, C3c])

        locX = np.zeros([l])
        locY = np.zeros([l])
        for i in range(l):
            locX[i] = Cc[0,i] + 0.5*Cc[2,i]
            locY[i] = np.sqrt(3)/2 * Cc[2,i]

    elif len(eigVal.shape)==3:
        l=eigVal.shape[1]
        l2=eigVal.shape[2]

        C1c = eigVal[0,:,:] - eigVal[1,:,:]
        C2c = 2*(eigVal[1,:,:] - eigVal[2,:,:])
        C3c = 3*eigVal[2,:,:] + 1
        Cc  = np.array([C1c, C2c, C3c])

        locX = np.zeros([l,l2])
        locY = np.zeros([l,l2])
        for i in range(l):
            for j in range(l2):
                locX[i,j] = Cc[0,i,j] + 0.5*Cc[2,i,j]
                locY[i,j] = np.sqrt(3)/2 * Cc[2,i,j]


        return np.array([locX, locY])

def baryToEigenvals(baryMap):
    # get the eigenvalues for a location on the barycentric map
    
    LHS = np.array([[1,-1,(3/2)],[0,0,(3*np.sqrt(3))/2],[1,1,1]])
    
    if len(baryMap.shape)==3:
        eigVals = np.zeros([3,baryMap.shape[1],baryMap.shape[2]])
        for i1 in range(baryMap.shape[1]):
            for i2 in range(baryMap.shape[2]):                
                RHS = np.transpose(np.array([baryMap[0,i1,i2]-(1/2),baryMap[1,i1,i2]-np.sqrt(3)/2,0]))
                eigVals[:,i1,i2] = np.dot(np.linalg.inv(LHS),RHS)
        
    elif len(baryMap.shape)==2:
        eigVals = np.zeros([3,baryMap.shape[1]])
        for i1 in range(baryMap.shape[1]):
            RHS = np.transpose(np.array([baryMap[0,i1]-(1/2),baryMap[1,i1]-np.sqrt(3)/2,0]))
            eigVals[:,i1] = np.dot(np.linalg.inv(LHS),RHS)
          
    return eigVals

def baryMap_discr(baryMap_RANS,baryMap_DNS):
    a = np.shape(baryMap_RANS)
    discr = np.zeros((2,a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            discr[0,i1,i2]=baryMap_DNS[0,i1,i2]-baryMap_RANS[0,i1,i2]
            discr[1,i1,i2]=baryMap_DNS[1,i1,i2]-baryMap_RANS[1,i1,i2]
    return discr

    
def baryMap_dist(baryMap_RANS,baryMap_DNS):
    a = np.shape(baryMap_RANS)
    dist = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            dist[i1,i2]= np.sqrt((baryMap_RANS[0,i1,i2]-baryMap_DNS[0,i1,i2])**2 + (baryMap_RANS[1,i1,i2]-baryMap_DNS[1,i1,i2])**2)
    return dist

def phi_dist(phi_discr, test_discr_phi):
    a = np.shape(phi_discr)
    dist = np.zeros((a[1],a[2]))
    for i1 in range(a[1]):
        for i2 in range(a[2]):    
            dist[i1,i2]= np.sqrt((phi_discr[0,i1,i2]- test_discr_phi[0,i1,i2])**2 + (phi_discr[1,i1,i2]- test_discr_phi[1,i1,i2])**2 + (phi_discr[2,i1,i2]- test_discr_phi[2,i1,i2])**2 )
    return dist
     
def loadData_avg(case, dataset):
    if case == 'PeriodicHills':
        print('function loadData_avg, case =', case)
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
    if case == 'SquareDuct':
        print('function loadData_avg, case =', case)
        with open(dataset, 'r') as f:
            reader = csv.reader(f)
            names = next(reader)
            ubulk = next(reader)
            #print(names)
            data_list = np.zeros([len(names), 100000])
            for i,row in enumerate(reader):
                if row:
                    data_list[:,i] = np.array([float(ii) for ii in row])
       # print(i)
        data_list = data_list[:,:i+1] 
        data = {}
        for j,var in enumerate(names):
            data[var] = data_list[j,:]
                  
        return data
    if case == 'ConvergingDivergingChannel':
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
        
        with open(dataset, 'rb') as f:
            
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
        
        data_list =  np.zeros([20, 44468*5])

        for i in range(20):
            for j in range(44467):
                for k in range(5):
                    data_list[i][k+j*5] = dataFile['vals'][i][j][k]
            
        return data_list

def interpDNSOnRANS(case, dataDNS, meshRANS):
    if case == 'PeriodicHills':
        print('function interpDNSOnRANS, case =', case)
        names = dataDNS.keys()    
        data = {}
        xy = np.array([dataDNS['x'], dataDNS['y']]).T
        for var in names:
            if not var=='x' and not var=='y':        
                data[var] = interp.griddata(xy, dataDNS[var], (meshRANS[0,:,:], meshRANS[1,:,:]), method='linear')
    
        return data
    if case == 'SquareDuct':
        print('function interpDNSOnRANS, case =', case)
        names = dataDNS.keys()    
        data = {}
        xy = np.array([dataDNS['Z'], dataDNS['Y']]).T
    #    print(xy)
    #    print(dataDNS['um'])
        for var in names:
            if not var=='Z' and not var=='Y':        
                data[var] = interp.griddata(xy, dataDNS[var], (meshRANS[0], meshRANS[1]), method='linear')
    #    print('shape xy: ' + str(xy.shape))
        return data
    if case == 'ConvergingDivergingChannel':
        print('function interpDNSOnRANS, case =', case)
        #names = dataDNS.keys()    
        data = {}
        xy = np.array([dataDNS[0], dataDNS[1]]).T
        for var in range(20):
            if not var==0 and not var==1:        
                data[var] = interp.griddata(xy, dataDNS[var], (meshRANS[0,:,:], meshRANS[1,:,:]), method='linear')
    
        return data
        
    
    
    

def eigenDecomposition(aij):
    #eigendecomposition of a tensor, returns the eigenvalues and eigenvectors 
    # which are sorted (lambda_1 >= lambda_2 >= lambda_3)
    # TODO: make exception for non-2D mesh
    eigVec = np.zeros([3,3,aij.shape[2],aij.shape[3]])
    eigVal = np.zeros([3,aij.shape[2],aij.shape[3]])
    
    for i1 in range(aij.shape[2]):
        for i2 in range(aij.shape[3]):
            #calculate eigenvalues and vectors
            eigDecomp = np.linalg.eig(aij[:,:,i1,i2])
            
            # get sorting indices (argsort), and flip to get l1<=l2<=l3
            sort_index = np.flip(np.argsort(eigDecomp[0]),axis=-1)
            sortVec = eigDecomp[1][:,sort_index]
            sortVal = eigDecomp[0][sort_index]
            
            eigVec[:,:,i1,i2] = (sortVec)
            eigVal[:,i1,i2] = (sortVal)
    return eigVal,eigVec
'''
def eigenvectorToEuler(eigVec):
    # get the euler angles (phi) for the eigenvectors (use to train the random forest)
    # TODO: make exception for non-2D mesh
    phi = np.zeros([3,eigVec.shape[2],eigVec.shape[3]])
   
    epsilon = 1e-6
    FOR1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #standard frame of reference
    
    for i1 in range(eigVec.shape[2]):
        for i2 in range(eigVec.shape[3]):
            
            ## positief assenstelsel
            if np.linalg.det(eigVec[:,:,i1,i2]) < 0:
                eigVec[0,2,i1,i2] = eigVec[0,2,i1,i2]
                eigVec[1,2,i1,i2] = eigVec[1,2,i1,i2]
                eigVec[2,2,i1,i2] = eigVec[2,2,i1,i2]
             
            R = eigVec[:,:,i1,i2]
            Zxy = np.sqrt(R[0,2]**2+R[1,2]**2)
            
            if Zxy > epsilon:
                phi[0,i1,i2] = np.arctan2(R[0,1]*R[1,2] - R[1,1]*R[0,2], R[0,0]*R[1,2] - R[1,0]*R[0,2]) #alpha (rotates the z-axis)
                phi[1,i1,i2] = np.arctan2(Zxy, R[2,2])
                phi[2,i1,i2] = -np.arctan2(-R[0,2],R[1,2]) #gamma (rotates the x-axis)
               
            else:
                phi[0,i1,i2] = 0
                phi[1,i1,i2] = 0. if R[2,2] > 0 else np.pi
                phi[2,i1,i2] = -np.arctan2(-R[2,1],R[0,0]) #gamma (rotates the x-axis)
        
    return phi
'''
## library implementation
def eigenvectorToEuler(eigVec):
    phi = np.zeros([3,eigVec.shape[2],eigVec.shape[3]])
    for i1 in range(eigVec.shape[2]):
        for i2 in range(eigVec.shape[3]):
            eulang = mat2euler(eigVec[:,:,i1,i2])
            
            phi[0,i1,i2] = eulang[0]
            phi[1,i1,i2] = eulang[1]
            phi[2,i1,i2] = eulang[2]
        
    return phi

## library implementation
def eulerToEigenvector(phi):
    ZXZ = np.zeros([3,3,phi.shape[1], phi.shape[2]])
    for i1 in range(phi.shape[1]):
        for i2 in range(phi.shape[2]):
            #rotation matrix:
            ZXZ[:,:,i1,i2] = euler2mat(phi[0, i1, i2], phi[1, i1, i2], phi[2, i1, i2])
    return ZXZ

''' Foute oorspronkelijke versie
def eulerToEigenvector(phi):
    # reconstruct eigenvector from given euler angles
    # TODO: make exception for non-2D mesh
    FOR1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #standard frame of reference
    eigVec = np.zeros([3,3,phi.shape[1],phi.shape[2]]) #initialize eigenvector
    for i1 in range(phi.shape[1]):
        for i2 in range(phi.shape[2]):
            #rotation matrix:
            Rx = np.array([[1,0,0],[0, np.cos(phi[2,i1,i2]), -np.sin(phi[2,i1,i2])],[0, np.sin(phi[2,i1,i2]), np.cos(phi[2,i1,i2])]])
            Ry = np.array([[np.cos(phi[1,i1,i2]),0,np.sin(phi[1,i1,i2])],[0,1,0],[-np.sin(phi[1,i1,i2]),0,np.cos(phi[1,i1,i2])]])
            Rz = np.array([[np.cos(phi[0,i1,i2]),-np.sin(phi[0,i1,i2]),0],[np.sin(phi[0,i1,i2]),np.cos(phi[0,i1,i2]),0],[0,0,1]])
            eigVec[:,:,i1,i2] = np.dot(Rz,np.dot(Ry,np.dot(Rx,FOR1)))
    return eigVec
'''
'''
#### niet wikipedia maar omgekeerd
def eulerToEigenvector(phi):
    # reconstruct eigenvector from given euler angles
    # TODO: make exception for non-2D mesh
   # FOR1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #standard frame of reference
    ZXZ = np.zeros([3,3,phi.shape[1], phi.shape[2]])
    
    for i1 in range(phi.shape[1]):
        for i2 in range(phi.shape[2]):
            #rotation matrix:
            ZXZ[0,0,i1,i2] = np.cos(phi[2,i1,i2]) * np.cos(phi[0,i1,i2]) - np.cos(phi[1,i1,i2])*np.sin(phi[2,i1,i2])*np.sin(phi[0,i1,i2])
            ZXZ[0,1,i1,i2] = - np.cos(phi[2,i1,i2])*np.sin(phi[0,i1,i2]) - np.cos(phi[1,i1,i2]) * np.cos(phi[0,i1,i2]) * np.sin(phi[2,i1,i2])
            ZXZ[0,2,i1,i2] = np.sin(phi[2,i1,i2]) * np.sin(phi[1,i1,i2])
            ZXZ[1,0,i1,i2] = np.cos(phi[0,i1,i2])*np.sin(phi[2,i1,i2]) + np.cos(phi[2,i1,i2]) * np.cos(phi[1,i1,i2]) * np.sin(phi[0,i1,i2])
            ZXZ[1,1,i1,i2] = np.cos(phi[2,i1,i2])*np.cos(phi[1,i1,i2])*np.cos(phi[0,i1,i2]) - np.sin(phi[2,i1,i2]) * np.sin(phi[0,i1,i2])
            ZXZ[1,2,i1, i2] = - np.cos(phi[2,i1,i1]) * np.sin(phi[1,i1,i1])
            ZXZ[2,0,i1, i2] = np.sin(phi[1,i1,i1]) * np.sin(phi[0,i1,i1])
            ZXZ[2,1,i1, i2] = np.cos(phi[0,i1,i1]) * np.sin(phi[1,i1,i1])
            ZXZ[2,2,i1, i2] = np.cos(phi[1,i1,i1])
    return ZXZ
'''
'''
#### wikipedia, lijkt niet goed te werken
def eulerToEigenvector(phi):
    # reconstruct eigenvector from given euler angles
    # TODO: make exception for non-2D mesh
   # FOR1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #standard frame of reference
    ZXZ = np.zeros([3,3,phi.shape[1], phi.shape[2]])
    
    for i1 in range(phi.shape[1]):
        for i2 in range(phi.shape[2]):
            #rotation matrix:
            ZXZ[0,0,i1,i2] = np.cos(phi[0,i1,i2]) * np.cos(phi[2,i1,i2]) - np.cos(phi[1,i1,i2])*np.sin(phi[0,i1,i2])*np.sin(phi[2,i1,i2])
            ZXZ[0,1,i1,i2] = - np.cos(phi[0,i1,i2])*np.sin(phi[2,i1,i2]) - np.cos(phi[1,i1,i2]) * np.cos(phi[2,i1,i2]) * np.sin(phi[0,i1,i2])
            ZXZ[0,2,i1,i2] = np.sin(phi[0,i1,i2]) * np.sin(phi[1,i1,i2])
            ZXZ[1,0,i1,i2] = np.cos(phi[2,i1,i2])*np.sin(phi[0,i1,i2]) + np.cos(phi[0,i1,i2]) * np.cos(phi[1,i1,i2]) * np.sin(phi[2,i1,i2])
            ZXZ[1,1,i1,i2] = np.cos(phi[0,i1,i2])*np.cos(phi[1,i1,i2])*np.cos(phi[2,i1,i2]) - np.sin(phi[0,i1,i2]) * np.sin(phi[2,i1,i2])
            ZXZ[1,2,i1, i2] = - np.cos(phi[0,i1,i1]) * np.sin(phi[1,i1,i1])
            ZXZ[2,0,i1, i2] = np.sin(phi[1,i1,i1]) * np.sin(phi[2,i1,i1])
            ZXZ[2,1,i1, i2] = np.cos(phi[2,i1,i1]) * np.sin(phi[1,i1,i1])
            ZXZ[2,2,i1, i2] = np.cos(phi[1,i1,i1])
    return ZXZ
'''
'''
#### hardcode versie 
def eulerToEigenvector(phi):
    # reconstruct eigenvector from given euler angles
    # TODO: make exception for non-2D mesh
   # FOR1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #standard frame of reference
    ZXZ = np.zeros([3,3,phi.shape[1], phi.shape[2]])
    
    for i1 in range(phi.shape[1]):
        for i2 in range(phi.shape[2]):
            #rotation matrix:
            ZXZ[0,0,i1,i2] = np.cos(phi[2,i1,i2]) * np.cos(phi[0,i1,i2]) - np.cos(phi[1,i1,i2])*np.sin(phi[2,i1,i2])*np.sin(phi[0,i1,i2])
            ZXZ[0,1,i1,i2] = -(- np.cos(phi[2,i1,i2])*np.sin(phi[0,i1,i2]) - np.cos(phi[1,i1,i2]) * np.cos(phi[0,i1,i2]) * np.sin(phi[2,i1,i2]))
            ZXZ[0,2,i1,i2] = -(np.sin(phi[2,i1,i2]) * np.sin(phi[1,i1,i2]))
            ZXZ[1,0,i1,i2] = (np.cos(phi[0,i1,i2])*np.sin(phi[2,i1,i2]) + np.cos(phi[2,i1,i2]) * np.cos(phi[1,i1,i2]) * np.sin(phi[0,i1,i2]))
            ZXZ[1,1,i1,i2] = -(np.cos(phi[2,i1,i2])*np.cos(phi[1,i1,i2])*np.cos(phi[0,i1,i2]) - np.sin(phi[2,i1,i2]) * np.sin(phi[0,i1,i2]))
            ZXZ[1,2,i1, i2] = -(- np.cos(phi[2,i1,i1]) * np.sin(phi[1,i1,i1]))
            ZXZ[2,0,i1, i2] = -(np.sin(phi[1,i1,i1]) * np.sin(phi[0,i1,i1]))
            ZXZ[2,1,i1, i2] = (np.cos(phi[0,i1,i1]) * np.sin(phi[1,i1,i1]))
            ZXZ[2,2,i1, i2] = np.cos(phi[1,i1,i1])
    return ZXZ
'''
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
    
