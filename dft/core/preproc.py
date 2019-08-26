import logging
from logging.config import fileConfig
import pathlib
rootpath = pathlib.Path(__file__).parent.parent
fileConfig(rootpath/'config'/'logging_config.ini')
logger = logging.getLogger()

import numpy as np
from scipy.interpolate import interp1d
from dft.util import fileextract as fe
from dft.util import gridsetup as gs

def condition(r, data, conditioning):
    for i in range(len(data)):
        if r[i] != 0.0:
            if conditioning == 1:
                data[i] = data[i]/(4*np.pi*r[i]**2)
            if conditioning == 2:
                data[i] = data[i]/r[i]
    return data

def interpolate(N, xyzs, refxyz, r, data, conditioning=None):
    if conditioning:
        data = condition(r, data, conditioning)
    if len(data) < len(r):
        rfunc = interp1d(r[:len(data)], data, kind='cubic')
        cutoff = r[len(data)-1]
    else:
        rfunc = interp1d(r, data, kind='cubic')
        cutoff = r[len(r)-1]
    datanew = np.zeros(N)
    for i in range(N):
        radsq = 0.0
        for ax in range(3):
            radsq += (xyzs[i][ax]-refxyz[ax])**2
        rad = np.sqrt(radsq)
        if rad <= cutoff:
            datanew[i] = rfunc(rad)
        else:
            datanew[i] = 0.0
    return datanew

def atomdata(N, xyzs, refxyz, pseudo):
    atom = {}
    atom['nc'] = interpolate(N, xyzs, refxyz, pseudo['r'], pseudo['nlcc'])
    atom['nv'] = interpolate(N, xyzs, refxyz, pseudo['r'], pseudo['rho'], 1)
    atom['vl'] = interpolate(N, xyzs, refxyz, pseudo['r'], pseudo['local'])
    betas = {}
    for k in pseudo['betas']:
        betas[k] = interpolate(N, xyzs, refxyz, pseudo['r'], pseudo['betas'][k], 2)
    atom['betas'] = betas
    q = {}
    for k in pseudo['q']:
        q[k] = interpolate(N, xyzs, refxyz, pseudo['r'], pseudo['q'][k], 1)
    atom['q'] = q
    wf = {}
    for k in pseudo['wf']:
        wf[k] = interpolate(N, xyzs, refxyz, pseudo['r'], pseudo['wf'][k], 2)
    atom['wf'] = wf
    atom['dion'] = pseudo['d']
    atom['Q'] = pseudo['Q']
    return atom

def setupatoms(N, xyzs, atoms):
    logger.info('Entered')
    atomsnew = []
    for species in atoms:
        pseudo = fe.extractpseudo(species+'.xml')
        for atom in atoms[species]:
            atomnew = atomdata(N, xyzs, atom['xyz'], pseudo)
            atomsnew.append(atomnew)
    logger.info('Exiting')
    return atomsnew

def totals(N, atoms):
    nc = np.zeros(N)
    nv = np.zeros(N)
    vl = np.zeros(N)
    for atom in atoms:
        nc += atom['nc']
        nv += atom['nv']
        vl += atom['vl']
    return nc, nv, vl
    
def run():
    logger.info('Entered')
    inp = fe.extractinputs()
    N, h, lin2grid, grid2lin, xyzs, neighs = gs.setup(inp['dim'], inp['div'])
    atoms = setupatoms(N, xyzs, inp['atoms'])
    nc, nv, vl = totals(N, atoms)
    logger.info('Exiting')
    return inp['div'], N, h, lin2grid, grid2lin, xyzs, neighs, atoms, nc, nv, vl

def run2():
    logger.info('Entered')
    inp = fe.extractinputs()
    N, h, lin2grid, grid2lin, xyzs, neighs = gs.setup(inp['dim'] , inp['div'])
    atoms = setupatoms(N, xyzs, inp['atoms'])
    nc, nv, vl = totals(N, atoms)
    logger.info('Exiting')
    return N, h, neighs, atoms, nc, nv, vl
