import os
import pickle
from wobbles.workflow.tabulate_pot import TabulatedPotential
import numpy as np

path_to_potentials = os.getcwd() + '/saved_potential_60/MW_'
pot_list = []
for idx in range(0, 2601):

    f = open(path_to_potentials + str(idx), 'rb')
    pot = pickle.load(f)
    f.close()
    pot_list.append(pot)

step = 0.01
nfw_normalizations = np.arange(0.1, 0.6 + step, step)
disk_min, disk_max, step = 0.35, 0.85, 0.01
disk_normalizations = np.arange(disk_min, disk_max+step, step)

tabpot = TabulatedPotential(pot_list, nfw_normalizations, disk_normalizations)

f = open(os.getcwd() + '/saved_potential_60' + '/tabulated_MWpot60', 'wb')
pickle.dump(tabpot, f)
f.close()