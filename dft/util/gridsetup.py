import logging
from logging.config import fileConfig
import pathlib
rootpath = pathlib.Path(__file__).parent.parent
fileConfig(rootpath/'config'/'logging_config.ini')
logger = logging.getLogger()

def count(div):
    N = div[0]*div[1]*div[2]
    return N

def spacings(dim, div):
    h = (dim[0]/div[0], dim[1]/div[1], dim[2]/div[2])
    return h

def gridmap(div):
    ##############################################################          
    ## Two-way map between point's grid index and linear index  ##
    ## tuple of keys <--> key                                   ##
    ##############################################################
    lin2grid = {}
    grid2lin = {}
    i = 0
    for xi in range(div[0]):
        for yi in range(div[1]):
            for zi in range(div[2]):
                lin2grid[i] = (xi, yi, zi)
                grid2lin[(xi, yi, zi)] = i
                i += 1
    return lin2grid, grid2lin

def coordinates(N, h, lin2grid):
    xyzs = {}
    for i in range(N):
        (nx, ny, nz) = lin2grid[i]
        xyzs[i] = (nx*h[0], ny*h[1], nz*h[2])
    return xyzs

def neighbours1(pos, count):
    ## Positions of first neighbours along an axis
    if pos == 0:
        return (count-1), (pos+1)
    if pos == (count-1):
        return (pos-1), 0
    else:
        return (pos-1), (pos+1)

def neighboursmap(lin2grid, grid2lin, div):
    ##################################################################            
    ## Map of linear index to indices of neighbours along each axis ##
    ## key --> tuple (axes) of tuple (left, right) of keys          ##
    ##################################################################
    neighs = {}
    for lini in lin2grid:
        gridi = lin2grid[lini]
        x = gridi[0]
        y = gridi[1]
        z = gridi[2]
        xjs = neighbours1(gridi[0], div[0])
        yjs = neighbours1(gridi[1], div[1])
        zjs = neighbours1(gridi[2], div[2])
        linxjs = (grid2lin[(xjs[0], y, z)], grid2lin[(xjs[1], y, z)])
        linyjs = (grid2lin[(x, yjs[0], z)], grid2lin[(x, yjs[1], z)])
        linzjs = (grid2lin[(x, y, zjs[0])], grid2lin[(x, y, zjs[1])])
        linjs = (linxjs, linyjs, linzjs)
        neighs[lini] = linjs
    return neighs

def setup(dim, div):
    logger.info('Entered')
    N = count(div)
    h = spacings(dim, div)
    lin2grid, grid2lin = gridmap(div)
    xyzs = coordinates(N, h, lin2grid)
    neighs = neighboursmap(lin2grid, grid2lin, div)
    logger.info('Exiting')
    return N, h, lin2grid, grid2lin, xyzs, neighs
