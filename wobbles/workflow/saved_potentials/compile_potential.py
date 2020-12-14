import os
import pickle
from wobbles.workflow.tabulate_pot import TabulatedPotential2D, TabulatedPotential3D
import numpy as np

path_to_potentials = os.getcwd() + '/saved_potentials/MW_'
pot_list = []

for idx in range(0, 20746):

    try:
        f = open(path_to_potentials + str(idx), 'rb')
        pot = pickle.load(f)
        f.close()
    except:
        f = open(path_to_potentials + str(idx-1), 'rb')
        pot = pickle.load(f)
        f.close()
    pot_list.append(pot)

step = 0.01
nfw_normalizations = np.loadtxt(os.getcwd() + '/saved_potentials/nfw_norms.txt')
disk_normalizations = np.loadtxt(os.getcwd() + '/saved_potentials/disk_norms.txt')
scale_heights = np.loadtxt(os.getcwd() + '/saved_potentials/scale_heights.txt')

tabpot = TabulatedPotential3D(pot_list, nfw_normalizations, disk_normalizations, scale_heights)

f = open(os.getcwd() + '/tabulated_MWpot_3D100', 'wb')
pickle.dump(tabpot, f)
f.close()