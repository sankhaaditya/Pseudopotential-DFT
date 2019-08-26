import pathlib
import logging
from logging.config import fileConfig
import numpy as np
from context import dft
from dft.util import fileextract as fe
from dft.core import preproc as pre
from dft.core import process as pro
##import matplotlib.pyplot as plt
from scipy import linalg as la
import pickle
from scipy import special as sp
import xml.etree.ElementTree as ET

rootpath = pathlib.Path(__file__).parent.parent
fileConfig(rootpath/'dft'/'config'/'logging_config.ini')
logger = logging.getLogger()

def extractpseudo():
    file = rootpath/'dft'/'pseudopotentials'/'cu.xml'
    pseudo = fe.extractpseudo(file)
    #logger.debug(pseudo['r'][612])
    return pseudo

def extractinputs():
    file = rootpath/'dft'/'input.xml'
    inputs = fe.extractinputs(file)
    #logger.debug(inputs)
    return inputs

def grid(dim, div):
    N = pre.count(div)
    h = pre.spacings(dim, div)
    lin2grid, grid2lin = pre.gridmap(div)
    neighs = pre.neighboursmap(lin2grid, grid2lin, div)
    coords = pre.coordinates(N, h, lin2grid)
    return N, h, lin2grid, grid2lin, neighs, coords

def potentials(div, N, h, neighs, atoms, nc, nv):
    n = np.random.rand(N)
    vxc = pro.xcpotential(N, h, neighs, n)
    logger.debug(vxc)
    vh = pro.hartreepotential(N, h, neighs, n)
    logger.debug(vh)
    vnl = pro.nonlocalpotential(N, atoms)
    logger.debug(vnl)

def runall():
    logger.info('Running all tests')
    div, N, h, lin2grid, grid2lin, xyzs, neighs, atoms, nc, nv, vl = pre.run()
    pro.run(div, N, h, lin2grid, grid2lin, neighs, atoms, nc, nv, vl)
    logger.info('Tests completed')

def checktotalcharge():
    occ = [2, 6, 8, 2, 0]
    l = [0, 1, 2, 0, 1]
    pseudo = fe.extractpseudo('cu.xml')
    wf = pseudo['wf']
    r = pseudo['r']
    print(r[575])
    rab = pseudo['rab']
    rho = pseudo['rho']
    betas = pseudo['betas']
    qs = pseudo['q']
    Qs = pseudo['Q']
    rhon = np.zeros(len(rho))
    prods = {}
    for w in wf:
        for b in betas:
            summ = 0.0
            for i in range(len(betas[b])):
                summ += wf[w][i]*betas[b][i]*rab[i]
            prods[(w, b)] = summ
    print(prods)
    summ = 0.0
    for i in range(len(r)):
        summ += rab[i]*wf[0][i]*wf[0][i]
    print(summ)
    summ += Qs[(1, 1)]*prods[(0, 1)]*prods[(0, 1)]
    summ += Qs[(1, 2)]*prods[(0, 1)]*prods[(0, 2)]
    summ += Qs[(1, 2)]*prods[(0, 2)]*prods[(0, 1)]
    summ += Qs[(2, 2)]*prods[(0, 2)]*prods[(0, 2)]
##    for Q in Qs:
##        s = 0.0
##        for i in range(len(r)):
##            s += qs[Q][i]*rab[i]
##        Qs[Q] = s
##        print(s)
##        if Q[0] == Q[1]:
##            summ += Qs[Q]*(prods[(0, Q[0])]*prods[(1, Q[1])])
##        else:
##            summ += Qs[Q]*(prods[(0, Q[0])]*prods[(1, Q[1])]+\
##                        prods[(0, Q[1])]*prods[(1, Q[0])])
    print(summ)
    for i in range(len(r)):
        for w in wf:
            rhon[i] += occ[w]*wf[w][i]**2
    for i in range(len(r)):
        for w in wf:
            rhon[i] += (qs[(l[w]*2+1,l[w]*2+1)][i]*prods[(w,l[w]*2+1)]*prods[(w, l[w]*2+1)]+\
             2*qs[(l[w]*2+1, l[w]*2+2)][i]*prods[(w, l[w]*2+1)]*prods[(w, l[w]*2+2)]+\
             qs[(l[w]*2+2, l[w]*2+2)][i]*prods[(w, l[w]*2+2)]*prods[(w, l[w]*2+2)])*occ[w]
    summrho = 0.0
    for i in range(len(r)):
        summrho += rhon[i]*rab[i]
    error = abs(summrho-18)/18*100
    assert error < 1.0

def checktotalchargegrid():
    #div, N, h, lin2grid, grid2lin, xyzs, neighs, atoms, nc, nv, vl = pre.run()
    # Loading d, N, h
    f = open('dNh.txt', 'rb')
    (d, N, h) = pickle.load(f)
    f.close()
    # Loading atoms
    f = open('atoms.txt', 'rb')
    atoms = pickle.load(f)
    f.close()
    vol = h[0]*h[1]*h[2]
    occ = [2, 6, 8, 2, 0]
    l = [0, 1, 2, 0, 1]
    wf = atoms[0]['wf']
    betas = atoms[0]['betas']
    qs = atoms[0]['q']
    rho = np.zeros(N)
    prods = {}
    logger.info('Done with pre run')
    for w in wf:
        for b in betas:
            summ = 0.0
            for i in range(len(betas[b])):
                summ += wf[w][i]*betas[b][i]*vol
            prods[(w, b)] = summ
    print(prods)
    logger.info('Starting basic addition')
    for i in range(N):
        for w in wf:
            rho[i] += occ[w]*wf[w][i]**2
    logger.info('Starting higher addition')
    for i in range(N):
        for w in wf:
            rho[i] += (qs[(l[w]*2+1,l[w]*2+1)][i]*prods[(w,l[w]*2+1)]*prods[(w, l[w]*2+1)]+\
             2*qs[(l[w]*2+1, l[w]*2+2)][i]*prods[(w, l[w]*2+1)]*prods[(w, l[w]*2+2)]+\
             qs[(l[w]*2+2, l[w]*2+2)][i]*prods[(w, l[w]*2+2)]*prods[(w, l[w]*2+2)])*occ[w]
    logger.info('Done higher addition')
    summrho = 0.0
    for i in range(N):
        summrho += rho[i]*vol
    error = abs(summrho-18)/18*100
    print(summrho)
    assert error < 1.0

def updatevl():
    file = rootpath/'dft'/'pseudopotentials'/'cu.xml'
    root = tree = ET.parse(file)
    r = fe.parse(root.find('PP_MESH').find('PP_R'))
    vl = fe.parse(root.find('PP_LOCAL'))
    f = open('xyzs.txt', 'rb')
    xyzs = pickle.load(f)
    f.close()
    vl = pre.interpolate(100**3,xyzs,(10,10,10),r,vl,2)
    f = open('vl.txt', 'wb')
    pickle.dump(vl, f)
    f.close()

def plotcheck():
    pseudo = fe.extractpseudo('cu')
    wf = pseudo['wf']
    r = pseudo['r']
    rab = pseudo['rab']
    rho = pseudo['rho']
    qs = pseudo['q']
    Qs = pseudo['Q']
    bs = pseudo['betas']
##    for q in qs:
##        print(q, Qs[q])
##        summ = 0.0
##        for i in range(len(r)):
##            summ += qs[q][i]*rab[i]
##        print(summ)
    for q in qs:
        plt.plot(r[:len(qs[q])], qs[q])
##    plt.plot(r, rho)
    plt.show()

def checkksequation():
    ## In Rydbergs

    en = -10.01984066000
    (l0,m0,i0) = (0,0,0)

##    # Extracting predata and saving
##    d, N, h, lin2grid, grid2lin, xyzs, neighs, atoms, nc, nv, vl = pre.run()
##    #N, h, neighs, atoms, nc, nv, vl = pre.run2()
##    f = open('dNh.txt', 'ab')
##    pickle.dump((d, N, h), f)
##    f.close()
##    del d, N, h
##    f = open('lin2grid.txt', 'ab')
##    pickle.dump(lin2grid, f)
##    f.close()
##    del lin2grid
##    f = open('grid2lin.txt', 'ab')
##    pickle.dump(grid2lin, f)
##    f.close()
##    del grid2lin
##    f = open('xyzs.txt', 'ab')
##    pickle.dump(xyzs, f)
##    f.close()
##    del xyzs
##    f = open('neighs.txt', 'ab')
##    pickle.dump(neighs, f)
##    f.close()
##    del neighs
##    f = open('atoms.txt', 'ab')
##    pickle.dump(atoms, f)
##    f.close()
##    del atoms
##    f = open('nc.txt', 'ab')
##    pickle.dump(nc, f)
##    f.close()
##    del nc
##    f = open('nv.txt', 'ab')
##    pickle.dump(nv, f)
##    f.close()
##    del nv
##    f = open('vl.txt', 'wb')
##    pickle.dump(vl, f)
##    f.close()
##    del vl

    # Loading d, N, h
    f = open('dNh.txt', 'rb')
    (d, N, h) = pickle.load(f)
    f.close()
    
    # Loading neighs
    f = open('neighs.txt', 'rb')
    neighs = pickle.load(f)
    f.close()
##
##    # Loading atoms
##    f = open('atoms.txt', 'rb')
##    atoms = pickle.load(f)
##    f.close()

    vol = h[0]*h[1]*h[2]

    # Extracting things from atoms and saving
    # Requires: atoms
##    qint = atoms[0]['Q']
##    f = open('qint.txt', 'ab')
##    pickle.dump(qint, f)
##    f.close()
##    del qint
##    dion = atoms[0]['dion']
##    f = open('dion.txt', 'ab')
##    pickle.dump(dion, f)
##    f.close()
##    del dion
##    betas = atoms[0]['betas']
##    f = open('betas.txt', 'ab')
##    pickle.dump(betas, f)
##    f.close()
##    del betas
##    q = atoms[0]['q']
##    f = open('q.txt', 'ab')
##    pickle.dump(q, f)
##    f.close()
##    del q
##    wf = atoms[0]['wf']
##    f = open('wf.txt', 'ab')
##    pickle.dump(wf, f)
##    f.close()
##    del wf
##    del atoms

    # Loading wf
    f = open('wf.txt', 'rb')
    wf = pickle.load(f)
    f.close()
    
    # Loading betas
    f = open('betas.txt', 'rb')
    betas = pickle.load(f)
    f.close()

    # Loading dion
    f = open('dion.txt', 'rb')
    dion = pickle.load(f)
    f.close()

    # Loading q
    f = open('q.txt', 'rb')
    q = pickle.load(f)
    f.close()

    # Loading qint
    f = open('qint.txt', 'rb')
    qint = pickle.load(f)
    f.close()

    # Loading harms
    f = open('harms.txt', 'rb')
    ha = pickle.load(f)
    f.close()

    # Calculating prods and saving
    # Requires: betas, wf
##    logger.info('prods')
##    p = {}
##    for l1 in range(3):
##        for m1 in range(-l1,l1+1):
##            if l1+3 in wf:
##                w = [wf[l1],wf[l1+3]]
##            else:
##                w = [wf[l1]]
##            for i in range(len(w)):
##                for l2 in range(3):
##                    for m2 in range(-l2,l2+1):
##                        b = [betas[2*l2+1],betas[2*l2+2]]
##                        for j in range(len(b)):
##                            s = 0.0
##                            for k in range(N):
##                                s += w[i][k]*np.conjugate(ha[(l1,m1)][k])*\
##                                     b[j][k]*ha[(l2,m2)][k]
##                            s = s.real*vol
##                            if abs(s) > 1.0e-12:
##                                p[(l1,m1,i,l2,m2,j)] = s
##                            else:
##                                p[(l1,m1,i,l2,m2,j)] = 0.0
##                            print((l1,m1,i,l2,m2,j),p[(l1,m1,i,l2,m2,j)])
##    del wf, betas
##    f = open('prods.txt', 'wb')
##    pickle.dump(p, f)
##    f.close()
##    del p

    # Loading prods
    f = open('prods.txt', 'rb')
    p = pickle.load(f)
    f.close()

    # Occupancy
    occ = {}
    occ[(0, 0, 0)] = 2
    occ[(0, 0, 1)] = 1
    occ[(1, -1, 0)] = 2
    occ[(1, -1, 1)] = 0
    occ[(1, 0, 0)] = 2
    occ[(1, 0, 1)] = 0
    occ[(1, 1, 0)] = 2
    occ[(1, 1, 1)] = 0
    occ[(2, -2, 0)] = 2
    occ[(2, -1, 0)] = 2
    occ[(2, 0, 0)] = 2
    occ[(2, 1, 0)] = 2
    occ[(2, 2, 0)] = 2

    # Calculating density and saving
    # Requires: wf, prods, harms
##    logger.info('density')
##    rho = np.zeros(N)
##    for l1 in range(3):
##        for m1 in range(-l1,l1+1):
##            if l1+3 in wf:
##                w = [wf[l1],wf[l1+3]]
##            else:
##                w = [wf[l1]]
##            for i in range(len(w)):
##                print((l1,m1,i))
##                for j in range(occ[(l1,m1,i)]):
##                    for k in range(N):
##                        rho[k] += (w[i][k]*abs(ha[(l1,m1)][k]))**2
##                    for l2 in range(3):
##                        for m2 in range(-l2,l2+1):
##                            b1 = 2*l2+1
##                            b2 = 2*l2+2
##                            rho += q[(b1,b1)]*p[(l1,m1,i,l2,m2,0)]*\
##                                   p[(l1,m1,i,l2,m2,0)]+\
##                                   2*q[(b1,b2)]*p[(l1,m1,i,l2,m2,0)]*\
##                                   p[(l1,m1,i,l2,m2,1)]+\
##                                   q[(b2,b2)]*p[(l1,m1,i,l2,m2,1)]*\
##                                   p[(l1,m1,i,l2,m2,1)]
##    f = open('nv.txt', 'wb')
##    pickle.dump(rho, f)
##    f.close()
##    del rho
       
    # Calculating Laplace matrix and saving
    # Requires: neighs
##    logger.info('lp')
##    lp = pro.laplace(N, h, neighs)
##    f = open('lp.txt', 'wb')
##    pickle.dump(lp, f)
##    f.close()
##    del lp
    
    # Loading lp
    f = open('lp.txt', 'rb')
    lp = pickle.load(f)
    f.close()

    # Calculating lpwf and saving
    # Requires: lp, wf, ha
##    logger.info('lpwf')
##    w = wf[l0+3*i0]
##    wh = np.zeros(N,dtype=np.complex_)
##    for k in range(N):
##        wh[k] = w[k]*ha[(l0,m0)][k]
##    lpwf = lp.dot(wh)
##    del lp, wf, ha
##    f = open('lpwf.txt', 'wb')
##    pickle.dump(lpwf, f)
##    f.close()
##    del lpwf

    # Loading nv
    f = open('nv.txt', 'rb')
    nv = pickle.load(f)
    f.close()

    # Loading nc
    f = open('nc.txt', 'rb')
    nc = pickle.load(f)
    f.close()

    # Calculating vh and saving
    # Requires: 
##    logger.info('vh')
##    vh = fft()
####    del nv
##    f = open('vh.txt', 'wb')
##    pickle.dump(vh, f)
##    f.close()
##    del vh

    # Loading vh
    f = open('vh.txt', 'rb')
    vh = pickle.load(f)
    f.close()

    # Calculating vhwf and saving
    # Requires: vh, harms, wf
##    logger.info('vhwf')
##    w = wf[l0+3*i0]
##    vhwf = np.zeros(N,dtype=np.complex_)
##    for k in range(N):
##        vhwf[k] += vh[k]*w[k]*ha[(l0,m0)][k]
##    del vh, ha, wf
##    f = open('vhwf.txt', 'wb')
##    pickle.dump(vhwf, f)
##    f.close()
##    del vhwf

    # Loading vhwf
    f = open('vhwf.txt', 'rb')
    vhwf = pickle.load(f)
    f.close()

    # Calculating vxc and saving
    # Requires: neighs, nv, nc
##    logger.info('vxc')
##    vxc = pro.xcpotential(N, h, neighs, nv+nc)
##    del neighs, nv, nc
##    f = open('vxc.txt', 'wb')
##    pickle.dump(vxc, f)
##    f.close()
##    del vxc

    # Loading vxc
    f = open('vxc.txt', 'rb')
    vxc = pickle.load(f)
    f.close()

    print(sum(vxc), max(vxc), min(vxc))

    # Calculating vxcwf and saving
    # Requires: vxc, wf
##    logger.info('vxcwf')
##    w = wf[l0+3*i0]
##    vxcwf = np.zeros(N,dtype=np.complex_)
##    for k in range(N):
##        vxcwf[k] += vxc[k]*w[k]*ha[(l0,m0)][k]
##    del vxc, wf
##    f = open('vxcwf.txt', 'ab')
##    pickle.dump(vxcwf, f)
##    f.close()
##    del vxcwf

    # Loading vxcwf
    f = open('vxcwf.txt', 'rb')
    vxcwf = pickle.load(f)
    f.close()

    # Loading lpwf
    f = open('lpwf.txt', 'rb')
    lpwf = pickle.load(f)
    f.close()

    # Loading vl
    f = open('vl.txt', 'rb')
    vl = pickle.load(f)
    f.close()
    
    # Calculating vlwf and saving
    # Requires: vl, wf, ha
##    logger.info('vlwf')
##    w = wf[l0+3*i0]
##    vlwf = np.zeros(N,dtype=np.complex_)
##    for k in range(N):
##        vlwf[k] = vl[k]*w[k]*ha[(l0,m0)][k]
##    del vl, wf, ha
##    f = open('vlwf.txt', 'wb')
##    pickle.dump(vlwf, f)
##    f.close()
##    del vlwf

    # Loading vlwf
    f = open('vlwf.txt', 'rb')
    vlwf = pickle.load(f)
    f.close()
    
    # Calculating vnlionwf and saving
    # Requires: prods, harms, dion, betas
##    logger.info('vnlionwf')
##    vnlionwf = np.zeros(N,dtype=np.complex_)
##    for l in range(3):
##        for m in range(-l,l+1):
##            b1 = 2*l+1
##            b2 = 2*l+2
##            bh1 = np.zeros(N,dtype=np.complex_)
##            bh2 = np.zeros(N,dtype=np.complex_)
##            for k in range(N):
##                bh1[k] = betas[b1][k]*ha[(l,m)][k]
##                bh2[k] = betas[b2][k]*ha[(l,m)][k]
##            vnlionwf += dion[(b1,b1)]*bh1*p[(l0,m0,i0,l,m,0)]+\
##                        dion[(b1,b2)]*bh1*p[(l0,m0,i0,l,m,1)]+\
##                        dion[(b1,b2)]*bh2*p[(l0,m0,i0,l,m,0)]+\
##                        dion[(b2,b2)]*bh2*p[(l0,m0,i0,l,m,1)]
##    del p, ha, dion, betas
##    f = open('vnlionwf.txt', 'wb')
##    pickle.dump(vnlionwf, f)
##    f.close()
##    del vnlionwf

    # Loading vnlionwf
    f = open('vnlionwf.txt', 'rb')
    vnlionwf = pickle.load(f)
    f.close()

    # Calculating dhxc
    # Requires: vxc, vh, q
##    logger.info('dhxc')
##    vhxc = 2*(vxc+vh) # Converting from Ha to Ry
##    del vxc
##    del vh
##    dhxc = {}
##    for k in q:
##        summ = 0.0
##        for i in range(N):
##            summ += q[k][i]*vhxc[i]
##        dhxc[k] = summ*vol
##    del vhxc
##    f = open('dhxc.txt', 'wb')
##    pickle.dump(dhxc, f)
##    f.close()
##    del dhxc

    # Loading dhxc
    f = open('dhxc.txt', 'rb')
    dhxc = pickle.load(f)
    f.close()

    # Calculating vnlhxcwf and saving
    # Requires: prods, harms dhxc, betas
##    logger.info('vnlhxcwf')
##    vnlhxcwf = np.zeros(N,dtype=np.complex_)
##    for l in range(3):
##        for m in range(-l,l+1):
##            b1 = 2*l+1
##            b2 = 2*l+2
##            bh1 = np.zeros(N,dtype=np.complex_)
##            bh2 = np.zeros(N,dtype=np.complex_)
##            for k in range(N):
##                bh1[k] = betas[b1][k]*ha[(l,m)][k]
##                bh2[k] = betas[b2][k]*ha[(l,m)][k]
##            vnlhxcwf += dhxc[(b1,b1)]*bh1*p[(l0,m0,i0,l,m,0)]+\
##                        dhxc[(b1,b2)]*bh1*p[(l0,m0,i0,l,m,1)]+\
##                        dhxc[(b1,b2)]*bh2*p[(l0,m0,i0,l,m,0)]+\
##                        dhxc[(b2,b2)]*bh2*p[(l0,m0,i0,l,m,1)]
##    del p, ha, dhxc, betas
##    f = open('vnlhxcwf.txt', 'wb')
##    pickle.dump(vnlhxcwf, f)
##    f.close()
##    del vnlhxcwf

    # Loading vnlhxcwf
    f = open('vnlhxcwf.txt', 'rb')
    vnlhxcwf = pickle.load(f)
    f.close()

    # Calculating olwf and saving
    # Requires: wf, prods, harms, qint, betas
##    logger.info('olwf')
##    olwf = np.zeros(N,dtype=np.complex_)
##    for l in range(3):
##        for m in range(-l,l+1):
##            b1 = 2*l+1
##            b2 = 2*l+2
##            bh1 = np.zeros(N,dtype=np.complex_)
##            bh2 = np.zeros(N,dtype=np.complex_)
##            for k in range(N):
##                bh1[k] = betas[b1][k]*ha[(l,m)][k]
##                bh2[k] = betas[b2][k]*ha[(l,m)][k]
##            olwf += qint[(b1,b1)]*bh1*p[(l0,m0,i0,l,m,0)]+\
##                    qint[(b1,b2)]*bh1*p[(l0,m0,i0,l,m,1)]+\
##                    qint[(b1,b2)]*bh2*p[(l0,m0,i0,l,m,0)]+\
##                    qint[(b2,b2)]*bh2*p[(l0,m0,i0,l,m,1)]
##    del wf, p, ha, qint, betas
##    f = open('olwf.txt', 'wb')
##    pickle.dump(olwf, f)
##    f.close()
##    del olwf

    # Loading olwf
    f = open('olwf.txt', 'rb')
    olwf = pickle.load(f)
    f.close()

##    print(la.norm(lpwf), la.norm(vlwf), la.norm(vnlionwf),\
##          la.norm(vnlhxcwf), la.norm(vhwf), la.norm(vxcwf))
    
    # Calculating lhs
    # Requires: lpwf, vlwf, vnlionwf, vnlhxcwf, vhwf, vxcwf
##    logger.info('lhs')
##    lhs = -lpwf+vlwf+vnlionwf+vnlhxcwf+2*vhwf+2*vxcwf
##    del lpwf, vlwf, vnlionwf, vnlhxcwf, vhwf, vxcwf
##    f = open('lhs.txt', 'wb')
##    pickle.dump(lhs, f)
##    f.close()
##    del lhs

    # Loading lhs
    f = open('lhs.txt', 'rb')
    lhs = pickle.load(f)
    f.close()

    # Calculating the wf dot the lhs
##    logger.info('final')
##    w = wf[l0+3*i0]
##    enl = 0.0
##    for k in range(N):
##        enl += w[k]*np.conjugate(ha[(l0,m0)][k])*lhs[k]
##    enl *= vol
##    print(enl)

    # Calculating rhs
    # Requires: olwf
##    logger.info('rhs')
##    rhs = e*olwf
##    del olwf
##    f = open('rhs.txt', 'wb')
##    pickle.dump(rhs, f)
##    f.close()
##    del rhs

    # Loading rhs
##    f = open('rhs.txt', 'rb')
##    rhs = pickle.load(f)
##    f.close()

    # Checking lhs and rhs
##    logger.info('check')
##    err = lhs+rhs
##    print(max(lhs), max(rhs))
##    print(min(lhs), min(rhs))
##    print(la.norm(lhs), la.norm(rhs), la.norm(err))

    # Total energy
    logger.info('total')
    tenl = 0.0
    tel = 0.0
    tek = 0.0
    teh = 0.0
    texc = 0.0
    for l1 in range(3):
        for m1 in range(-l1,l1+1):
            if l1+3 in wf:
                w = [wf[l1],wf[l1+3]]
            else:
                w = [wf[l1]]
            for i in range(len(w)):
                print((l1,m1,i))
                for j in range(occ[(l1,m1,i)]):
                    for l2 in range(3):
                        for m2 in range(-l2,l2+1):
                            b1 = 2*l2+1
                            b2 = 2*l2+2
                            tenl += dion[(b1,b1)]*p[(l1,m1,i,l2,m2,0)]*\
                                   p[(l1,m1,i,l2,m2,0)]+\
                                   2*dion[(b1,b2)]*p[(l1,m1,i,l2,m2,0)]*\
                                   p[(l1,m1,i,l2,m2,1)]+\
                                   dion[(b2,b2)]*p[(l1,m1,i,l2,m2,1)]*\
                                   p[(l1,m1,i,l2,m2,1)]
                    wh = np.zeros(N,dtype=np.complex_)
                    for k in range(N):
                        wh[k] = w[i][k]*ha[(l1,m1)][k]
                    tek += -lp.dot(wh).dot(np.conjugate(wh)).real*vol
                    for k in range(N):
                        tel += vl[k]*abs(wh[k])**2*vol
    for k in range(N):
        teh += 2*vh[k]*nv[k]*vol
        texc += 2*vxc[k]*nv[k]*vol
    te = tenl+tel+tek+teh+texc
    print(tenl,tel,tek,teh,texc,te)

    

def verify():
    rem = list(map(float, open("rem.txt", "r").read().split()))
    rem2 = list(map(float, open("rem2.txt", "r").read().split()))
    error = 0.0
    for i in range(len(rem)):
        if rem[i] != 0.0:
            error += abs(rem[i]-rem2[i])/abs(rem[i])
    error /= len(rem)

def iterative():
    N, h, neighs, atoms, nc, nv, vl = pre.run2()
    del atoms, nc, vl
    lp = pro.laplace(N, h, neighs)
    del neighs
    A = lp.tolil()
    del lp
    B = np.zeros(N)
    for i in range(N):
        A[0, i] = 0.0
        B[i] = -4*np.pi*nv[i]
    del nv
    A[0, 0] = 1.0
    A = A.toarray()
    B[0] = 0.0
    Au = la.triu(A)
    Al = la.tril(A)
    Ad = np.diag(np.diag(A))
##    for i in range(N):
##        for j in range(N):
##            if A[i][j] != Au[i][j]+Al[i][j]-Ad[i][j]:
##                print(i, j)
    x = np.zeros(N)
    T = 1000
    RJ = np.dot(la.inv(Ad), A)+np.identity(N)
    RGS = -np.dot(la.inv(Al-Ad), Au)
    val, vec = la.eig(RGS)
    maxx = 0.0
    for i in range(len(val)):
        if abs(val[i]) > maxx:
            maxx = abs(val[i])
    print(maxx)
##    for t in range(T):
##        x=np.dot(RJ, x)-np.dot(la.inv(Ad), B)
##        print(la.norm(x))

def fft():
    import pyfftw
    #f = pyfftw.empty_aligned((100,100,100),dtype='complex128')
    #f = pyfftw.n_byte_align_empty((20,20,20),16, dtype='complex128')
    #f[:] = np.random.randn(*f.shape)
    #fftf=pyfftw.interfaces.numpy_fft.fftn(f)
    #ifftf=pyfftw.interfaces.numpy_fft.ifftn(fftf)
##    d, N, h, lin2grid, grid2lin, xyzs, neighs, atoms, nc, nv, vl = pre.run()
    #f = np.random.rand(20, 20, 20)
    f = open('dNh.txt', 'rb')
    (d, N, h) = pickle.load(f)
    f.close()
    f = open('neighs.txt', 'rb')
    neighs = pickle.load(f)
    f.close()
    f = open('nv.txt', 'rb')
    nv = pickle.load(f)
    f.close()
##    f = open('nc.txt', 'rb')
##    nc = pickle.load(f)
##    f.close()
    f = open('grid2lin.txt', 'rb')
    grid2lin = pickle.load(f)
    f.close()
    #lp = pro.laplace(N, h, neighs)
##    del lin2grid, xyzs, neighs, atoms, vl
    a = 20
    d = d[0]
    n = np.zeros([d,d,d])
    #vxc = pro.xcpotential(N, h, neighs, nv+nc)
    #print(la.norm(vxc))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                n[i][j][k] = nv[grid2lin[(i,j,k)]]
    ng = pyfftw.interfaces.numpy_fft.fftn(n)#/a**3
    del n
    def so(i):
        if i <= d/2:
            return i
        else:
            return d-i
    vg = np.zeros([d,d,d])
    for i in range(d):
        for j in range(d):
            for k in range(d):
                if i == 0 and j == 0 and k == 0:
                    vg[i][j][k] = 0.0
                    continue
                Gsq = (2*np.pi/a)**2*(so(i)**2+so(j)**2+so(k)**2)
                vg[i][j][k] = 4*np.pi/Gsq*ng[i][j][k]
    del ng
    v = pyfftw.interfaces.numpy_fft.ifftn(vg)#/a**3
    del vg
    vh = np.zeros(d**3)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                vh[grid2lin[(i,j,k)]] = v[i][j][k]
    del grid2lin, v
    return vh
##    fa = np.zeros(N)
##    lpvh = lp.dot(vh)
##    nvm = 4*np.pi*nv
##    for i in range(N):
##        fa[i] = (lpvh[i])+(nvm[i])
##    print(la.norm(lpvh), la.norm(nvm), la.norm(fa), la.norm(vh))

def sphericalharmonics():
    ## Testing spherical harmonics 
    f = open('dNh.txt', 'rb')
    (d, N, h) = pickle.load(f)
    f.close()
    f = open('xyzs.txt', 'rb')
    xyzs = pickle.load(f)
    f.close()
    ## HARDCODING
    ref = 10.0
    l = 1
    m = 1
    theta = np.zeros(N)
    phi = np.zeros(N)
    for k in xyzs:
        (x, y, z) = xyzs[k]
        r = np.sqrt((x-ref)**2+(y-ref)**2+(z-ref)**2)
        ## theta and phi are opposite to usual
        ## theta varies from -pi to pi, phi varies from 0 to pi
        if r != 0.0:
            phi[k] = np.arccos((z-ref)/r)
        else:
            phi[k] = 0.0
        if (x-ref) != 0.0:
            if (x-ref) >= 0.0:
                theta[k] = np.arctan((y-ref)/(x-ref))
            else:
                if (y-ref) >= 0.0:
                    theta[k] = np.arctan((y-ref)/(x-ref))+np.pi
                else:
                    theta[k] = np.arctan((y-ref)/(x-ref))-np.pi
        else:
            if (y-ref) >= 0.0:
                theta[k] = np.pi/2
            else:
                theta[k] = -np.pi/2
    print(max(theta), min(theta), max(phi), min(phi))
    ## Generating harmonics for required l, m on 100 cube grid and saving
    harms = {}
    for l in range(3):
        for m in range(-l, l+1):
            h = {}
            for k in xyzs:                
                ## Spherical harmonics already include Cordon-Shortley phase
                h[k] = sp.sph_harm(m, l, theta[k], phi[k])
            harms[(l,m)] = h
    f = open('harms.txt', 'wb')
    pickle.dump(harms, f)
    f.close()

def new():
    f = open('dNh.txt', 'rb')
    (d, N, h) = pickle.load(f)
    f.close()
    vol = h[0]*h[1]*h[2]
    del h
    f = open('harms.txt', 'rb')
    h = pickle.load(f)
    f.close()
    f = open('betas.txt', 'rb')
    betas = pickle.load(f)
    f.close()
    f = open('wf.txt', 'rb')
    wf = pickle.load(f)
    f.close()
    f = open('q.txt', 'rb')
    q = pickle.load(f)
    f.close()
    f = open('qint.txt', 'rb')
    qint = pickle.load(f)
    f.close()    
    p = {}
    for l1 in range(3):
        for m1 in range(-l1,l1+1):
            if l1+3 in wf:
                w = [wf[l1],wf[l1+3]]
            else:
                w = [wf[l1]]
            for i in range(len(w)):
                for l2 in range(3):
                    for m2 in range(-l2,l2+1):
                        b = [betas[2*l2+1],betas[2*l2+2]]
                        for j in range(len(b)):
                            s = 0.0
                            for k in range(N):
                                s += w[i][k]*np.conjugate(h[(l1,m1)][k])*\
                                     b[j][k]*h[(l2,m2)][k]
                            s = s.real*vol
                            if abs(s) > 1.0e-12:
                                p[(l1,m1,i,l2,m2,j)] = s
                            else:
                                p[(l1,m1,i,l2,m2,j)] = 0.0
                            print((l1,m1,i,l2,m2,j),p[(l1,m1,i,l2,m2,j)])
    f = open('p2.txt', 'wb')
    pickle.dump(p, f)
    f.close()
    for l1 in range(3):
        for m1 in range(-l1,l1+1):
            if l1+3 in wf:
                w = [wf[l1],wf[l1+3]]
            else:
                w = [wf[l1]]
            for i in range(len(w)):
                print((l1,m1,i))
                ch = 0.0
                for k in range(N):
                    ch += (w[i][k]*abs(h[(l1,m1)][k]))**2
                ch *= vol
                print(ch)
                for l2 in range(3):
                    for m2 in range(-l2,l2+1):
                        b1 = 2*l2+1
                        b2 = 2*l2+2
                        ch += qint[(b1,b1)]*p[(l1,m1,i,l2,m2,0)]*\
                               p[(l1,m1,i,l2,m2,0)]+\
                               2*qint[(b1,b2)]*p[(l1,m1,i,l2,m2,0)]*\
                               p[(l1,m1,i,l2,m2,1)]+\
                               qint[(b2,b2)]*p[(l1,m1,i,l2,m2,1)]*\
                               p[(l1,m1,i,l2,m2,1)]
                        print(ch)

def testxc():
    import pylibxc
    f = open('neighs.txt', 'rb')
    neighs = pickle.load(f)
    f.close()
    f = open('dNh.txt', 'rb')
    (d,N,h) = pickle.load(f)
    f.close()
    f = open('nc.txt', 'rb')
    nc = pickle.load(f)
    f.close()
    f = open('nv.txt', 'rb')
    nv = pickle.load(f)
    f.close()
    n = nc+nv
    grads, sigmas = pro.densitygradient(N, h, neighs, n)
##    func = pylibxc.LibXCFunctional("GGA_XC_PBE", "unpolarized")
##    inp = {'rho': n, 'sigma': sigmas}
##    ret = func.compute(inp)
##    zk = ret['zk'][0]
##    vrho = ret['vrho'][0]
##    vsigmas = ret['vsigma'][0]
##    tmp = {}
##    for i in range(N):
##        tmp[i] = vsigmas[i]*grads[i]
##    vxc = np.zeros(N)
##    for i in range(N):
##        tmpres = 0.0
##        for ax in range(3):
##            lval = tmp[neighs[i][ax][0]][ax]
##            rval = tmp[neighs[i][ax][1]][ax]
##            tmpres += pro.centraldifference(lval, rval, h[ax])
##        vxc[i] = zk[i]+vrho[i]-2*tmpres
##    print(min(vxc), max(vxc), la.norm(vxc))
    func = pylibxc.LibXCFunctional("LDA_XC_KSDT", "unpolarized")
    inp = {'rho': n}
    ret = func.compute(inp)
    zk = ret['zk'][0]
    vrho = ret['vrho'][0]
    vxc = np.zeros(N)
    for i in range(N):
        vxc[i] = zk[i]+vrho[i]
    print(sum(vxc), max(vxc), min(vxc))

##    nn = np.zeros(2*N)
##    sn = np.zeros(3*N)
##    for i in range(N):
##        nn[i] = n[i]/2
##        nn[i+N] = n[i]/2
##        sn[i] = sigmas[i]/4
##        sn[i+N] = sigmas[i]/4
##        sn[i+2*N] = sigmas[i]/4
##    inp = {'rho': nn, 'sigma': sn}
    

##    f = open('vxc.txt', 'wb')
##    pickle.dump(vrho,f)
##    f.close()
    
if __name__ == '__main__':
    #runall()
    #checktotalcharge()
    #checktotalchargegrid()
    #updatevl()
    #plotcheck()
    checkksequation()
    #iterative()
    #fft()
    #sphericalharmonics()
    #new()
    #testxc()
