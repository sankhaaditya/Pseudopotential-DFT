import pathlib
import logging
from logging.config import fileConfig
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sla
from scipy import linalg as la
import pylibxc

rootpath = pathlib.Path(__file__).parent.parent
fileConfig(rootpath/'config'/'logging_config.ini')
logger = logging.getLogger()

def centraldifference(lvalue, rvalue, h):
    ## 2nd order central difference approximation for 1st derivative
    derivative = (rvalue-lvalue)/(2*h)
    return derivative

def finitedifferenceterms(h):
    mterm = -2*(1/h[0]**2+1/h[1]**2+1/h[2]**2)
    nterms = (1/h[0]**2, 1/h[1]**2, 1/h[2]**2)
    return mterm, nterms

def densitygradient(N, h, neighs, n):
    ##################################################
    ## Calculate the density gradient at each point
    ## key --> array (axes) of floats
    ## Calculate sigmas as gradient squared
    ## key --> float
    ##################################################
    grads = {}
    sigmas = np.zeros(N)
    for i in range(N):
        grad = np.zeros(3)
        sigma = 0.0
        for ax in range(3):
            lval = n[neighs[i][ax][0]]
            rval = n[neighs[i][ax][1]]
            grad[ax] = centraldifference(lval, rval, h[ax])
            sigma += grad[ax]*grad[ax]
        grads[i]= grad
        sigmas[i] = (sigma)
    return grads, sigmas

def xcpotential(N, h, neighs, n):
    logger.info('Entered')
    grads, sigmas = densitygradient(N, h, neighs, n)
    func = pylibxc.LibXCFunctional("GGA_XC_PBE1W", "unpolarized")
    inp = {'rho': n, 'sigma': sigmas}
    ret = func.compute(inp)
    vrho = ret['vrho'][0]
    vsigmas = ret['vsigma'][0]
    tmp = {}
    for i in range(N):
        tmp[i] = vsigmas[i]*grads[i]
    vxc = np.zeros(N)
    for i in range(N):
        tmpres = 0.0
        for ax in range(3):
            lval = tmp[neighs[i][ax][0]][ax]
            rval = tmp[neighs[i][ax][1]][ax]
            tmpres += centraldifference(lval, rval, h[ax])
        vxc[i] = vrho[i]-2*tmpres
    logger.info('Exiting')
    return vxc

def laplace(N, h, neighs):
    logger.info('Entered')
    row = np.zeros(N*7)
    col = np.zeros(N*7)
    data = np.zeros(N*7)
    count = 0
    mterm, nterms = finitedifferenceterms(h)
    def addtosparse(i, j, entry):
        nonlocal count
        row[count] = i
        col[count] = j
        data[count] = entry
        count += 1
    for i in range(N):
        addtosparse(i, i, mterm)
        for ax in range(3):
            for j in range(2):
                addtosparse(i, neighs[i][ax][j], nterms[ax])
    lp = csc_matrix((data, (row, col)), shape=(N, N))
    logger.info('Exiting')
    return lp

##def hartreepotential(N, nv):

##def hartreepotential(N, lp, n):
##    A = lp.tolil()
##    del lp
##    B = np.zeros(N)
##    for i in range(N):
##        A[0, i] = 0.0
##        B[i] = -4*np.pi*n[i]
##    del n
##    A[0, 0] = 1.0
##    A = A.tocsc()
##    B[0] = 0.0
##    logger.info('Solving Poisson')
##    vh = sla.spsolve(A, B)
##    logger.info('Done')
##    return vh

def updated(N, h, vhxc, atoms):
    logger.info('Entered')
    vol = h[0]*h[1]*h[2]
    for atom in atoms:
        if 'd' not in atom:
            atom['d'] = {}
        for k in atom['q']:
            f = np.zeros(N)
            for i in range(N):
                f[i] = 2*vhxc[i]*atom['q'][k][i]   ## Factor 2 for Ha to Ry
            res = vol*sum(f)
            if k in atom['dion']:
                res += atom['dion'][k]
            atom['d'][k] = res
    logger.info('Exiting')
    return atoms

def nonlocalpotential(N, h, atoms):
    logger.info('Entered')
    vol = h[0]*h[1]*h[2]
    vnl = np.zeros([N, N])
    for atom in atoms:
        ds = atom['d']
        b = atom['betas']
        for d in ds:
            for i in range(N):
                for j in range(N):
                    if d[0] == d[1]:
                        vnl[i][j] += vol**2*ds[d]*b[d[0]][i]*b[d[1]][j]
                    else:
                        vnl[i][j] += vol**2*ds[d]*(b[d[0]][i]*b[d[1]][j]+\
                                      b[d[1]][i]*b[d[0]][j])
    logger.info('Exiting')
    return vnl

def overlapmatrix(N, h, atoms):
    logger.info('Entered')
    vol = h[0]*h[1]*h[2]
    S = np.zeros([N, N])
    for i in range(N):
        S[i][i] = vol
    for atom in atoms:
        Qs = atom['Q']
        b = atom['betas']
        for Q in Qs:
            for i in range(N):
                for j in range(N):
                    if Q[0] == Q[1]:
                        S[i][j] += vol**2*Qs[Q]*b[Q[0]][i]*b[Q[1]][j]
                    else:
                        S[i][j] += vol**2*Qs[Q]*(b[Q[0]][i]*b[Q[1]][j]+\
                                      b[Q[1]][i]*b[Q[0]][j])
    logger.info('Exiting')
    return S

def hamiltonian(N, h, vl, vhxc, vnl, lp):
    logger.info('Entered')
    #(row, col, data) = laplace
    vol = h[0]*h[1]*h[2]
    H = np.zeros([N, N])
    for i in range(N):
        H[i][i] += vol*(vhxc[i]+0.5*vl[i])
        for j in range(N):
            H[i][j] += (0.5*vnl[i][j]-vol*0.5*lp[i][j])
##    count = 0
##    for i in range(N):
##        if count < len(bounds) and i == bounds[count]:
##            for j in range(N):
##                S[i][j] = 0.0
##            count += 1
##        else:
##            for j in range(N):
##                H[i][j] = 0.5*vnl[i][j]     ## Convert Rydberg to Hartree
##            H[i][i] = vhxc[i]+0.5*vl[i]   ## Same here (*0.5)
##    for i in range(len(data)):
##        H[row[i]][col[i]] = -0.5*data[i]
    logger.info('Exiting')
    return H

def sort(eigenValues, eigenVectors):
    idx = eigenValues.argsort()#[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors

def density(N, h, atoms, H, S):
    logger.info('Entered')
    vol = h[0]*h[1]*h[2]
##    val, vec = la.eig(H, S)
##    val, vec = sort(val, vec)
##    vec = np.transpose(vec)
    s, U = np.linalg.eig(S)
    s_mhalf = np.diag(s**(-0.5))
    S_mhalf = np.dot(U, np.dot(s_mhalf, U.T))
    X = S_mhalf
    F_prime = np.dot(X.T, np.dot(H, X))
    E, C_prime = np.linalg.eigh(F_prime)
    C = np.dot(X, C_prime)
    vec = np.transpose(C)
    rho1 = np.zeros(N)
    rho2 = np.zeros(N)
    prods = {}
    vN = 9  ## HARDCODED!!
    for v in range(vN):   
        for a in range(len(atoms)):
            betas = atoms[a]['betas']
            for b in betas:
                summ = 0.0
                for i in range(N):
                    summ += vec[v][i]*betas[b][i]
                prods[(a, b, v)] = vol*summ     
    for v in range(vN):
        for i in range(N):
            rho1[i] += 2*vec[v][i]**2
    for v in range(vN):
        for a in range(len(atoms)):
            qs = atoms[a]['q']
            for q in qs:
                if q[0] == q[1]:
                    rho2[i] += 2*qs[q][i]*prods[(a, q[0], v)]*\
                               prods[(a, q[1], v)]
                else:
                    rho2[i] += 2*2*qs[q][i]*prods[(a, q[0], v)]*\
                               prods[(a, q[1], v)]
    summ1 = 0.0
    summ2 = 0.0
    for i in range(N):
        summ1 += rho1[i]
        summ2 += rho2[i]
    print(summ1*vol, summ2*vol, summ1*vol+summ2*vol)
    nv = rho1+rho2
    logger.info('Exiting')
    return nv
                
def run(div, N, h, lin2grid, grid2lin, neighs, atoms, nc, nv, vl):
    ########################################################################   
    ## Start with valence density nv                                      ##
    ## Calculate Laplace matrix using geometry                            ##
    ## Calculate XC potential vxc using nv+nc                             ##
    ## Calculate Hartree potential vh using nv                            ##
    ## Calculate dhxc using vxc+vh, q                                     ##
    ## Calculate nonlocal nuclear potential vnl using dion+dhxc, betas    ##
    ## Calculate overlap matrix and add vxc, vh, vnl to form H matrix     ##                             ##
    ## Solve generalized eigenvalue problem                               ##
    ########################################################################
##    laplace, bounds = laplacematrix(div, N, h, lin2grid, grid2lin, neighs)
    lp = laplace(N, h, neighs)
    S = overlapmatrix(N, h, atoms)
    for t in range(10):
        n = nv+nc
        vxc = xcpotential(N, h, neighs, n)
##        vh = hartreepotential2(N, laplace, bounds, nv)
        vh = hartreepotential(N, lp, nv)
        vhxc = vh+vxc
        atoms = updated(N, h, vhxc, atoms)
        print(atoms[0]['dion'], atoms[0]['d'])
        vnl = nonlocalpotential(N, h, atoms)
        logger.debug('Local:\n'+str(min(vl))+'  '+str(max(vl))+'\n')
        logger.debug('XC:\n'+str(min(vxc))+'  '+str(max(vxc))+'\n')
        logger.debug('Hartree:\n'+str(min(vh))+'  '+str(max(vh))+'\n')
        maxx = []
        minn = []
        for i in range(len(vnl)):
            maxx.append(max(vnl[i]))
            minn.append(min(vnl[i]))
        logger.debug('Nonlocal:\n'+str(min(minn))+'  '+str(max(maxx))+'\n')
##        if t == 0:
##            S = overlapmatrix(N, h, atoms)
##        H, S = hamiltonian(N, vl, vhxc, vnl, laplace, bounds, S)
        H = hamiltonian(N, h, vl, vhxc, vnl, lp)
        nv = density(N, h, atoms, H, S)
##    count = 0
##    summ = 0.0
##    maxx = 0.0
##    minn = 0.0
##    for i in range(N):
##        for j in range(N):
##            if vnl[i][j] > maxx:
##                maxx = vnl[i][j]
##            if vnl[i][j] < minn:
##                minn = vnl[i][j]
##            summ += vnl[i][j]
##            if vnl[i][j] == 0.0:
##                count += 1
##    print(minn, maxx, summ/N**2)
##    count = 0
##    for i in range(N):
##        for j in range(N):
##            if S[i][j] == 0.0:
##                count += 1
##    print(count)
    #return vxc, vh, vnl





##def laplacematrix(div, N, h, lin2grid, grid2lin, neighs):
##    logger.info('Entered')
##    row = []
##    col = []
##    data = []
##    bounds = []
##    def addtosparse(i, j, v):
##        row.append(i)
##        col.append(j)
##        data.append(v)
##    mterm, nterms = finitedifferenceterms(h)
##    for key in range(N):
##        (i, j, k) = lin2grid[key]
##        if i == 0 and j == 0 and k == 0:
##            addtosparse(key, key, 1.0)
##            addtosparse(key, grid2lin[((1, 1, 1))], 1/8)
##            addtosparse(key, grid2lin[((1, 1, div[2]-1))], 1/8)
##            addtosparse(key, grid2lin[((1, div[1]-1, 1))], 1/8)
##            addtosparse(key, grid2lin[((1, div[1]-1, div[2]-1))], 1/8)
##            addtosparse(key, grid2lin[((div[0]-1, 1, 1))], 1/8)
##            addtosparse(key, grid2lin[((div[0]-1, 1, div[2]-1))], 1/8)
##            addtosparse(key, grid2lin[((div[0]-1, div[1]-1, 1))], 1/8)
##            addtosparse(key, grid2lin[((div[0]-1, div[1]-1, div[2]-1))], 1/8)
##            bounds.append(key)
##        elif i == 0 and j == 0 and k != 0:
##            addtosparse(key, key, 1.0)
##            addtosparse(key, grid2lin[((1, 1, k))], 1/4)
##            addtosparse(key, grid2lin[((1, div[1]-1, k))], 1/4)
##            addtosparse(key, grid2lin[((div[0]-1, 1, k))], 1/4)
##            addtosparse(key, grid2lin[((div[0]-1, div[1]-1, k))], 1/4)
##            bounds.append(key)
##        elif i == 0 and j != 0 and k == 0:
##            addtosparse(key, key, 1.0)
##            addtosparse(key, grid2lin[((1, j, 1))], 1/4)
##            addtosparse(key, grid2lin[((1, j, div[2]-1))], 1/4)
##            addtosparse(key, grid2lin[((div[0]-1, j, 1))], 1/4)
##            addtosparse(key, grid2lin[((div[0]-1, j, div[2]-1))], 1/4)
##            bounds.append(key)
##        elif i != 0 and j == 0 and k == 0:
##            addtosparse(key, key, 1.0)
##            addtosparse(key, grid2lin[((i, 1, 1))], 1/4)
##            addtosparse(key, grid2lin[((i, 1, div[2]-1))], 1/4)
##            addtosparse(key, grid2lin[((i, div[1]-1, 1))], 1/4)
##            addtosparse(key, grid2lin[((i, div[1]-1, div[2]-1))], 1/4)
##            bounds.append(key)
##        elif i == 0 and j != 0 and k != 0:
##            addtosparse(key, key, 1.0)
##            addtosparse(key, grid2lin[((1, j, k))], 1/2)
##            addtosparse(key, grid2lin[((div[0]-1, j, k))], 1/2)
##            bounds.append(key)
##        elif i != 0 and j == 0 and k != 0:
##            addtosparse(key, key, 1.0)
##            addtosparse(key, grid2lin[((i, 1, k))], 1/2)
##            addtosparse(key, grid2lin[((i, div[1]-1, k))], 1/2)
##            bounds.append(key)
##        elif i != 0 and j != 0 and k == 0:
##            addtosparse(key, key, 1.0)
##            addtosparse(key, grid2lin[((i, j, 1))], 1/2)
##            addtosparse(key, grid2lin[((i, j, div[2]-1))], 1/2)
##            bounds.append(key)
##        else:
##            addtosparse(key, key, mterm)
##            for ax in range(3):
##                for num in range(2):
##                    addtosparse(key, neighs[key][ax][num], nterms[ax])
##    logger.info('Exiting')
##    return (row, col, data), bounds            
##
##def hartreepotential2(N, laplace, bounds, n):
##    logger.info('Entered')
##    (row, col, data) = laplace
##    A = csc_matrix((data, (row, col)), shape=(N, N))
##    B = np.zeros(N)
##    count = 0
##    for i in range(N):
##        if count < len(bounds) and i == bounds[count]:
##            B[i] = 0.0
##            count += 1
##        else:
##            B[i] = -4*np.pi*n[i]
##    vh = la.solve(A.todense(), B)
##    logger.info('Exiting')
##    return vh
