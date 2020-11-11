import os
import pickle
from wobbles.workflow.tabulate_pot import TabulatedPotential
import numpy as np

path_to_potentials = './saved_potential_mwpot/MW_'
pot_list = []
for idx in range(0, 672):

    f = open(path_to_potentials + str(idx), 'rb')
    pot = pickle.load(f)
    f.close()
    pot_list.append(pot)
step = 0.01
disk_min, disk_max = 0.45, 0.75
nfw_normalizations = np.arange(0.15, 0.45 + step, step)
disk_normalizations = np.linspace(disk_min, disk_max, 21)
tabpot = TabulatedPotential(pot_list, nfw_normalizations, disk_normalizations)

f = open('tabulated_MWpot', 'wb')
pickle.dump(tabpot, f)
f.close()