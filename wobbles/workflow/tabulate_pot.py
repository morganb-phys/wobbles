import numpy as np
from galpy.potential import NFWPotential
from galpy.potential import MiyamotoNagaiPotential
from galpy.potential import PowerSphericalPotentialwCutoff
from wobbles.potential_extension import PotentialExtension
import pickle
import sys
import os

class TabulatedPotential(object):

    def __init__(self, potential_extension_list, param_1_values, param_2_values):

        """

        :param potential_extension_list:
        :param param_1_values: the parameter iterated over first in a double for loop
        :param param_2_values: the parameter iterated over second in a double for loop
        """
        self.potential_extension_list = potential_extension_list

        p1, p2 = np.meshgrid(param_2_values, param_1_values)

        self._p1_step = param_2_values[1] - param_2_values[0]
        self._p2_step = param_1_values[1] - param_1_values[0]
        self._p1, self._p2 = p1.ravel(), p2.ravel()

        self._p1min, self._p1max = min(param_1_values), max(param_1_values)
        self._p2min, self._p2max = min(param_2_values), max(param_2_values)

    def _min_index(self, p1, p2):

        # note the reversed order
        d_param1 = (p1 - self._p2)**2/self._p2_step**2
        d_param2 = (p2 - self._p1)**2/self._p1_step**2
        d = np.sqrt(d_param1 + d_param2)
        return np.argmin(d)

    def evaluate(self, param_1_value, param_2_value):

        assert param_1_value >= self._p1min and param_1_value <= self._p1max
        assert param_2_value >= self._p2min and param_2_value <= self._p2max

        index = self._min_index(param_1_value, param_2_value)
        return self.potential_extension_list[index]
#
#
# z_min_max_kpc = 2.
# vz_min_max_kmsec = 120
# phase_space_N = 60
#
# step = 0.01
# nfw_normalizations = np.arange(0.1, 0.5 + step, step)
#
# disk_min, disk_max = 0.35, 1.
# step = 0.01
# disk_normalizations = np.arange(disk_min, disk_max + step, step)
#
# np.savetxt('./saved_potentials/nfw_norms.txt', nfw_normalizations)
# np.savetxt('./saved_potentials/disk_norms.txt', disk_normalizations)
#
# counter = 1
# from time import time
# t0 = time()
# ntot = len(nfw_normalizations) * len(disk_normalizations)
# print('ntotal: ', ntot)
# pot_list = []
# for nfw_norm in nfw_normalizations:
#
#     for disk_norm in disk_normalizations:
#
#         fname = './saved_potentials/saved_potential_mwpot/MW_' + str(counter-1)
#
#         if counter != int(sys.argv[1]):
#             continue
#         if not os.path.exists(fname):
#             galactic_potential = [PowerSphericalPotentialwCutoff(normalize=0.05, alpha=1.8,
#                                                                  rc=1.9 / 8.),
#                                   MiyamotoNagaiPotential(a=3. / 8., b=0.28 / 8., normalize=disk_norm),
#                                   NFWPotential(a=2., normalize=nfw_norm)]
#
#             potential_local = PotentialExtension(galactic_potential, z_min_max_kpc, vz_min_max_kmsec, phase_space_N)
#
#             f = open(fname, 'wb')
#             pickle.dump(potential_local, f)
#             f.close()
#         else:
#             f = open(fname, 'rb')
#             potential_local = pickle.load(f)
#             f.close()
#
#         pot_list.append(potential_local)
#
#         counter += 1
#
#         if counter%10 == 0:
#             t = (time() - t0)/60
#             rate = (counter-1) / t
#             nremaining = ntot - counter
#             print('min remaining: ', np.round(nremaining/rate, 2))
#
# fname = './saved_potentials/tabulated_MWpot60'
# tabulated_potential = TabulatedPotential(pot_list, nfw_normalizations, disk_normalizations)
#
# f = open(fname, 'wb')
# pickle.dump(tabulated_potential, f)
# f.close()