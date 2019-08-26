import numpy as np
from scipy.interpolate import interp1d
from util import file

length = 5.0
div = 10
pos = 2.5
spacing = length/div

pseudo = file.extractpseudo("./pseudopotentials/cu.xml")
r = pseudo['r']
local = pseudo['local']
beta = pseudo['beta']
wf = pseudo['wf']
d = pseudo['d']
print(d)
for i in range(0, 6):
    for j in range(len(beta[i+1]), len(r)):
        beta[i+1].append(0.0)
    
wffunc = interp1d(r, wf, kind='cubic')

cr = np.zeros(div*div*div)
cwf = np.zeros(div*div*div)
index = 0
for i in range(0, div):
    for j in range(0, div):
        for k in range(0, div):
            x = spacing*i
            y = spacing*j
            z = spacing*k
            cr[index] = np.power((x-pos)**2+(y-pos)**2+(z-pos)**2, 0.5)
            cwf[index] = wffunc(cr[index])
            index += 1

cbetas = {}           
for key in beta:
    betafunc = interp1d(r, beta[key], kind='cubic')
    cbeta = np.zeros(div*div*div)
    index = 0
    for i in range(0, div):
        for j in range(0, div):
            for k in range(0, div):
                cbeta[index] = betafunc(cr[index])
                index += 1
    cbetas[key] = cbeta

vnl = np.zeros(div*div*div)
for key1 in beta:
    for key2 in beta:
        if (key1, key2) in d:
            index = 0
            add = 0.0
            for i in range(0, div):
                for j in range(0, div):
                    for k in range(0, div):
                        add += cwf[index]*cbetas[key1][index]
                        index += 1
            vnl += d[(key1, key2)]*add*cbetas[key2]
print(vnl)
