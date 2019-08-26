import logging
from logging.config import fileConfig
import pathlib
import sys
sys.path.append('..')
import xml.etree.ElementTree as ET
from dft.properties.constants import *
rootpath = pathlib.Path(__file__).parent.parent
fileConfig(rootpath/'config'/'logging_config.ini')
logger = logging.getLogger()

def xmlroot(file):
    try:
        tree = ET.parse(file)
    except:
        logger.error('Unable to parse file: %s', file)
    root = tree.getroot()
    return root

def split(attrib, dtype=float):
    strlist = attrib.split()
    data = tuple(map(dtype, strlist))
    return data

def parseatoms(element):
    atoms = {}
    for atom in element.findall('atom'):
        xyz = split(atom.get('coordinates'))
        frac =  float(atom.get('fraction'))
        species = atom.get('species')
        if species not in atoms:
            atoms[species] = []
        atoms[species].append({
            'xyz': xyz,
            'frac': frac
            })
    return atoms
                            
def extractinputs():
    ######################################################
    ## Extract the following from the input xml file:   ##
    ##   lattice constants - tuple, len 3               ##
    ##   atoms - list of atoms                          ##
    ##   mesh fineness - list of number of divisions    ##
    ######################################################
    logger.info('Entered')
    filename = 'input'+EXTENSION_XML
    root = xmlroot(filename)
    inputs = {}
    inputs['dim'] = split(root.find('lattice').get('dimensions'))
    inputs['div'] = split(root.find('lattice').get('divisions'), int)
    inputs['atoms'] = parseatoms(root.find('atoms'))
    logger.info('Exiting')
    return inputs

def parse(element, offset = 0):
    strlist = element.text.split()
    if offset == 0:
        data = list(map(float, strlist))
        return data
    else:
        predata = []
        for i in range(offset):
            predata.append(strlist[0])
            strlist.pop(0)
        data = list(map(float, strlist))
        return predata, data

def parsebetas(element):
    betas = {}
    for sub in element:
        predata, data = parse(sub, PP_BETA_OFFSET)
        index = int(predata[PP_BETA_I])
        betas[index] = data
    return betas

def parsed(element):
    d = {}
    strlist = element.text.split()
    for i in range(PP_DIJ_OFFSET, len(strlist), PP_DIJ_SKIP):
        index = (int(strlist[i+PP_DIJ_I]), int(strlist[i+PP_DIJ_J]))
        d[index] = float(strlist[i+PP_DIJ_VALUE])
    return d

def parseq(element):
    q = {}
    Q = {}
    for sub in element:
        predata, data = parse(sub, PP_QIJ_OFFSET)
        index = (int(predata[PP_QIJ_I]), int(predata[PP_QIJ_J]))
        Q[index] = float(predata[PP_QIJ_QINT])
        q[index] = data
    return q, Q

def parsewf(element):
    strlist = element.text.split()
    wf = {}
    for key in range(5):
        data = []
        for i in range((4+867)*key+4, (4+867)*(key+1)):   #HARDCODED FOR CU
            data.append(float(strlist[i]))
        wf[key] = data
    return wf
            
def extractpseudo(species):
    logger.info('Entered')
    filename = species#+EXTENSION_XML
    file = rootpath/'pseudopotentials'/filename
    root = xmlroot(file)
    pseudo = {}
    pseudo['r'] = parse(root.find(PP_MESH).find(PP_R))
    pseudo['rab'] = parse(root.find(PP_MESH).find(PP_RAB))
    pseudo['nlcc'] = parse(root.find(PP_NLCC))
    pseudo['local'] = parse(root.find(PP_LOCAL))
    pseudo['betas'] = parsebetas(root.find(PP_NONLOCAL).findall(PP_BETA))
    pseudo['d'] = parsed(root.find(PP_NONLOCAL).find(PP_DIJ))
    pseudo['q'], pseudo['Q'] = parseq(root.find(PP_NONLOCAL).findall(PP_QIJ))
    pseudo['rho'] = parse(root.find(PP_RHOATOM))
    pseudo['wf'] = parsewf(root.find(PP_PSWFC))
    logger.info('Exiting')
    return pseudo
