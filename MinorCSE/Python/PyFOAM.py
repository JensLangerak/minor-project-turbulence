#===============================================================================
# S U B R O U T I N E S
#===============================================================================
from __future__ import division
import numpy as np


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

def getRANSVector(case, time, var):
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


def calcFracAnisotropy(tau):
    if len(tau.shape)==3:
        l=tau.shape[2]

        eigVal = np.zeros([3,l])
        for i in range(l):
            a,b=np.linalg.eig(tau[:,:,i])
            eigVal[:,i]=sorted(a, reverse=True)
        #eigValMean = eigVal.mean(0)
        FA = np.zeros([l])
        for i in range(l):
#            FA[i] = np.sqrt(3/2.) * np.sqrt( (eigVal[0,i] - eigValMean[i])**2. + (eigVal[1,i] - eigValMean[i])**2. + (eigVal[2,i] - eigValMean[i])**2.) / np.sqrt(eigVal[0,i]**2. + eigVal[1,i]**2. + eigVal[2,i]**2.)
#            FA[i] = np.sqrt(1/2.) * np.sqrt( (eigVal[0,i] - eigVal[1,i])**2. + (eigVal[1,i] - eigVal[2,i])**2. + (eigVal[2,i] - eigVal[0,i])**2.) / np.sqrt(eigVal[0,i]**2. + eigVal[1,i]**2. + eigVal[2,i]**2.)
            FA[i] = np.sqrt(1/2.) * np.sqrt((eigVal[0,i] - eigVal[1,i])**2. + (eigVal[1,i] - eigVal[2,i])**2. + (eigVal[2,i] - eigVal[0,i])**2.) / np.sqrt(eigVal[0,i]**2. + eigVal[1,i]**2. + eigVal[2,i]**2.)


    elif len(tau.shape)==4:
        l=tau.shape[2]
        l2=tau.shape[3]
        print (l,l2)

        eigVal = np.zeros([3,l,l2])
        for i in range(l):
            for j in range(l2):
                a,b=np.linalg.eig(tau[:,:,i,j])
                eigVal[:,i,j]=sorted(a, reverse=True)

        print (l, l2)
        #eigValMean = eigVal.mean(0)
        FA = np.zeros([l,l2])
        for i in range(l):
            for j in range(l2):
#                FA[i,j] = np.sqrt(3/2.) * np.sqrt( (eigVal[0,i,j] - eigValMean[i,j])**2. + (eigVal[1,i,j] - eigValMean[i,j])**2. + (eigVal[2,i,j] - eigValMean[i,j])**2.) / np.sqrt(eigVal[0,i,j]**2. + eigVal[1,i,j]**2. + eigVal[2,i,j]**2.)
                FA[i,j] = np.sqrt(1/2.) * np.sqrt( (eigVal[0,i,j] - eigVal[1,i,j])**2. + (eigVal[2,i,j] - eigVal[1,i,j])**2. + (eigVal[2,i,j] - eigVal[0,i,j])**2.) / np.sqrt(eigVal[0,i,j]**2. + eigVal[1,i,j]**2. + eigVal[2,i,j]**2.)

    return FA

def calcInvariant(eigVal):
    if len(eigVal.shape)==2:
        l=eigVal.shape[1]

        II = np.zeros([l])
        III = np.zeros([l])
        for i in range(l):
            II[i] = 2.*(eigVal[0,i]**2. + eigVal[0,i] * eigVal[1,i] + eigVal[1,i]**2.)
            III[i] = -3*eigVal[0,i]*eigVal[1,i]*(eigVal[0,i] + eigVal[1,i])

    elif len(eigVal.shape)==3:
        l=eigVal.shape[1]
        l2=eigVal.shape[2]

        II = np.zeros([l,l2])
        III = np.zeros([l,l2])
        for i in range(l):
            for j in range(l2):
                II[i,j] = 2.*(eigVal[0,i,j]**2. + eigVal[0,i,j] * eigVal[1,i,j] + eigVal[1,i,j]**2.)
                III[i,j] = -3*eigVal[0,i,j]*eigVal[1,i,j]*(eigVal[0,i,j] + eigVal[1,i,j])

    return II,III


def calcInitialConditions(U, turbulenceIntensity, turbLengthScale, nu, d, D):

    k       = 1.5 * (U*turbulenceIntensity)**2.0
    epsilon = 0.16 * k**1.5 / turbLengthScale
    omega   = 1.8 * np.sqrt(k) / turbLengthScale
    #omega_farfield = U/D
    omega_wall = 10 * 6 * nu / (0.0705 * d**2)
    #omega_wall_wilcox = 6 / (0.0708 * yPlus_wilcox**2)
    nuTilda = np.sqrt(1.5)*U*turbulenceIntensity*turbLengthScale
    nuTilda_NASA = 3*nu
    nut_NASA = 3*nu*(3**3)/(3**3 + 7.1**3)
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


def getInvariants(Tensors):
    # get the invariants (lambda_1 ... lambda_k) for a set of tensors (e.g. strain/rotation rate tensors)
    # TODO: add cases where more tensors are used for input
    #shape of the results:
    a = np.shape(Tensors[0])
    
    if np.shape(Tensors)[0] == 2: #only using grad(U)
        S = Tensors[0]
        R = Tensors[1]
        invariants = np.zeros([6,a[2],a[3]])
        for i1 in range(a[2]):
            for i2 in range(a[3]):
                inv1 = np.trace( np.dot(S[:,:,i1,i2],S[:,:,i1,i2]) )
                inv2 = np.trace( np.dot(R[:,:,i1,i2],R[:,:,i1,i2]) )
                inv3 = np.trace( np.dot( S[:,:,i1,i2],np.dot(S[:,:,i1,i2],S[:,:,i1,i2])) )
                inv4 = np.trace( np.dot( R[:,:,i1,i2],np.dot(R[:,:,i1,i2],S[:,:,i1,i2])) )
                inv5 = np.trace( np.dot( R[:,:,i1,i2],np.dot(R[:,:,i1,i2],np.dot(S[:,:,i1,i2],S[:,:,i1,i2]))) )
                inv6_a = np.dot( R[:,:,i1,i2],np.dot(S[:,:,i1,i2],S[:,:,i1,i2]))
                inv6_b = np.dot(R[:,:,i1,i2],np.dot(R[:,:,i1,i2],np.dot(S[:,:,i1,i2],inv6_a)))
                inv6 = np.trace(inv6_b)
                invariants[:,i1,i2] = [inv1,inv2,inv3,inv4,inv5,inv6]

    return invariants

def getInvariants2(Tensors,scale=False,cap=3.0):
    # get the invariants (lambda_1 ... lambda_k) for a set of tensors (e.g. strain/rotation rate tensors)
    # scaling is included (0 mean, 1 std), which includes a cap which can be set to remove outliers
    # TODO: add cases where more tensors are used for input
    
    #shape of the results:
    a = np.shape(Tensors[0])
    
    if np.shape(Tensors)[0] == 2: #only using grad(U)
        S = Tensors[0]
        R = Tensors[1]
        invariants = np.zeros([6,a[2],a[3]])
        for i1 in range(a[2]):
            for i2 in range(a[3]):
                inv1 = np.trace( np.dot(S[:,:,i1,i2],S[:,:,i1,i2]) )
                inv2 = np.trace( np.dot(R[:,:,i1,i2],R[:,:,i1,i2]) )
                inv3 = np.trace( np.dot( S[:,:,i1,i2],np.dot(S[:,:,i1,i2],S[:,:,i1,i2])) )
                inv4 = np.trace( np.dot( R[:,:,i1,i2],np.dot(R[:,:,i1,i2],S[:,:,i1,i2])) )
                inv5 = np.trace( np.dot( R[:,:,i1,i2],np.dot(R[:,:,i1,i2],np.dot(S[:,:,i1,i2],S[:,:,i1,i2]))) )
                inv6_a = np.dot( R[:,:,i1,i2],np.dot(S[:,:,i1,i2],S[:,:,i1,i2]))
                inv6_b = np.dot(R[:,:,i1,i2],np.dot(R[:,:,i1,i2],np.dot(S[:,:,i1,i2],inv6_a)))
                inv6 = np.trace(inv6_b)
                invariants[:,i1,i2] = [inv1,inv2,inv3,inv4,inv5,inv6]
                
        std_inv = np.zeros(invariants.shape[0])
        mu_inv = np.zeros(invariants.shape[0])
        if scale == True:

            
            for i3 in range(invariants.shape[0]):
                # process consists of two parts: normalizing and removing outliers,
                # and normalizing and keeping the std/mean for later processing
                std_temp = np.std(invariants[i3,:,:])
                mu_temp = np.mean(invariants[i3,:,:])
                
                #normalize
                invariants[i3,:,:] = (invariants[i3,:,:] - mu_temp) / std_temp
                
                #cap
                invariants[i3,:,:][invariants[i3,:,:]> cap] = cap
                invariants[i3,:,:][invariants[i3,:,:]< -cap] = -cap
                
                #de-normalize
                invariants[i3,:,:] = (invariants[i3,:,:]*std_temp) + mu_temp
                
                # data with outliers removed: again get the mean and std
                std_inv[i3] = np.std(invariants[i3,:,:])
                mu_inv[i3] = np.mean(invariants[i3,:,:])
                
                #normalize the data with no outliers
                invariants[i3,:,:] = (invariants[i3,:,:] - mu_inv[i3]) / std_inv[i3]
                
            #print(std_inv,mu_inv)
    return invariants, std_inv, mu_inv 

def getTensorBasis(S,R,scale=False):
    #return the tensor basis from Pope(1975) for a strain rate/rotation rate tensor
    TensorBasis = np.zeros([3,3,10,S.shape[2],S.shape[3]])
    scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
    
    for i1 in range(S.shape[2]):
        for i2 in range(S.shape[3]):
            S_temp = S[:,:,i1,i2]
            R_temp = R[:,:,i1,i2]
            TensorBasis[:,:,0,i1,i2] = S_temp
            TensorBasis[:,:,1,i1,i2] = np.dot(S_temp,R_temp) - np.dot(R_temp,S_temp)
            TensorBasis[:,:,2,i1,i2] = np.dot(S_temp,S_temp) - (1/3)*np.eye(3)*np.trace(np.dot(S_temp,S_temp))
            TensorBasis[:,:,3,i1,i2] = np.dot(R_temp,R_temp) - (1/3)*np.eye(3)*np.trace(np.dot(R_temp,R_temp))
            TensorBasis[:,:,4,i1,i2] = np.dot(R_temp,np.dot(S_temp,S_temp)) - np.dot(S_temp,np.dot(S_temp,R_temp))
            TensorBasis[:,:,5,i1,i2] = (np.dot(R_temp,np.dot(R_temp,S_temp)) + np.dot(S_temp,np.dot(R_temp,R_temp)) 
            - (2/3)*np.eye(3)*np.trace(np.dot(S_temp,np.dot(R_temp,R_temp))))
            TensorBasis[:,:,6,i1,i2] = (np.dot(R_temp,np.dot(S_temp,np.dot(R_temp,R_temp))) - 
                       np.dot(R_temp,np.dot(R_temp,np.dot(S_temp,R_temp))))
            TensorBasis[:,:,7,i1,i2] = (np.dot(S_temp,np.dot(R_temp,np.dot(S_temp,S_temp))) - 
                       np.dot(S_temp,np.dot(S_temp,np.dot(R_temp,S_temp))))
            TensorBasis[:,:,8,i1,i2] = (np.dot(R_temp,np.dot(R_temp,np.dot(S_temp,S_temp))) +
                       np.dot(S_temp,np.dot(S_temp,np.dot(R_temp,R_temp))) - 
                       (2/3)*np.eye(3)*np.trace(np.dot(S_temp,np.dot(S_temp,np.dot(R_temp,R_temp)))) )
            TensorBasis[:,:,9,i1,i2] = (np.dot(R_temp,np.dot(S_temp,np.dot(S_temp,np.dot(R_temp,R_temp)))) - 
                       np.dot(R_temp,np.dot(R_temp,np.dot(S_temp,np.dot(S_temp,R_temp)))) )
            
            if scale==True:
                for i3 in range(10):
                    TensorBasis[:, :, i3, i1, i2] /= scale_factor[i3]
    return TensorBasis
     

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



def eigenDecomposition(aij):
    #eigendecomposition of a tensor, returns the eigenvalues and eigenvectors 
    # which are sorted (lambda_1 >= lambda_2 >= lambda_3)
    # TODO: make exception for non-2D mesh
    eigVec = np.zeros([3,3,aij.shape[2],aij.shape[3]])
    eigVal = np.zeros([3,3,aij.shape[2],aij.shape[3]])
    
    for i1 in range(aij.shape[2]):
        for i2 in range(aij.shape[3]):
            #calculate eigenvalues and vectors
            eigDecomp = np.linalg.eig(aij[:,:,i1,i2])
            
            # get sorting indices (argsort), and flip to get l1<=l2<=l3
            sort_index = np.flip(np.argsort(eigDecomp[0]),axis=-1)
            sortVec = eigDecomp[1][:,sort_index]
            sortVal = eigDecomp[0][sort_index]
            
            eigVec[:,:,i1,i2] = (sortVec)
            eigVal[:,:,i1,i2] = np.diag((sortVal))
    return eigVal,eigVec


def eigenvectorToEuler(eigVec):
    # get the euler angles (phi) for the eigenvectors (use to train the random forest)
    # TODO: make exception for non-2D mesh
    phi = np.zeros([3,eigVec.shape[2],eigVec.shape[3]])
    
    FOR1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #standard frame of reference
    for i1 in range(eigVec.shape[2]):
        for i2 in range(eigVec.shape[3]):
            R = np.dot(np.linalg.inv(FOR1),eigVec[:,:,i1,i2])
            phi[0,i1,i2] = np.arctan2(R[1,0],R[0,0]) #alpha (rotates the z-axis)
            phi[1,i1,i2] = np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
            phi[2,i1,i2] = np.arctan2(R[2,1],R[2,2]) #gamma (rotates the x-axis)
    return phi



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