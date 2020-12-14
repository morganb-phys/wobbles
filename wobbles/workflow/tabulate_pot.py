import numpy as np
from galpy.potential import NFWPotential
from galpy.potential import MiyamotoNagaiPotential
from galpy.potential import PowerSphericalPotentialwCutoff
from wobbles.potential_extension import PotentialExtension
import pickle
import sys
import os

class TabulatedPotential3D(object):

    def __init__(self, potential_extension_list, param_1_values, param_2_values, param_3_values):

        """

        :param potential_extension_list:
        :param param_1_values: the parameter iterated over first in a triple for loop
        :param param_2_values: the parameter iterated over second in a triple for loop
        :param param_3_values: the parameter iterated over third in a triple for loop
        """
        self.potential_extension_list = potential_extension_list

        shape = (len(param_1_values) * len(param_2_values) * len(param_3_values), 4)
        coordinates = np.empty(shape)

        count = 0
        for p1 in param_1_values:
            for p2 in param_2_values:
                for p3 in param_3_values:
                    coordinates[count, :] = np.array([p1, p2, p3, count])
                    count += 1

        self._coordinates = coordinates
        self._p1_step = param_3_values[1] - param_3_values[0]
        self._p2_step = param_2_values[1] - param_2_values[0]
        self._p3_step = param_1_values[1] - param_1_values[0]

        self._p1min, self._p1max = np.round(min(param_1_values), 2), np.round(max(param_1_values), 2)
        self._p2min, self._p2max = np.round(min(param_2_values), 2), np.round(max(param_2_values), 2)
        self._p3min, self._p3max = np.round(min(param_3_values), 2), np.round(max(param_3_values), 2)

    @property
    def param_ranges(self):

        return [self._p1min, self._p1max], [self._p2min, self._p2max], [self._p3min, self._p3max]

    def _min_index(self, p1, p2, p3):

        d_param1 = (p1 - self._coordinates[:, 0]) ** 2 / self._p1_step ** 2
        d_param2 = (p2 - self._coordinates[:, 1]) ** 2 / self._p2_step ** 2
        d_param3 = (p3 - self._coordinates[:, 2]) ** 2 / self._p3_step ** 2
        d_param = d_param1 + d_param2 + d_param3
        d = np.sqrt(d_param)
        idx = int(np.argmin(d))
        return int(self._coordinates[idx, -1])

    def evaluate(self, param_1_value, param_2_value, param_3_value):

        assert param_1_value >= self._p1min and param_1_value <= self._p1max
        assert param_2_value >= self._p2min and param_2_value <= self._p2max
        assert param_3_value >= self._p3min and param_3_value <= self._p3max

        index = self._min_index(param_1_value, param_2_value, param_3_value)
        return self.potential_extension_list[index]

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

    @property
    def param_ranges(self):
        return [self._p1min, self._p1max], [self._p2min, self._p2max]

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

# import numpy as np
# from galpy.potential import NFWPotential
# from galpy.potential import MiyamotoNagaiPotential
# from galpy.potential import PowerSphericalPotentialwCutoff
# from wobbles.potential_extension import PotentialExtension
# import pickle
# import sys
# import os
#
# z_min_max_kpc = 2.
# vz_min_max_kmsec = 120
# phase_space_N = 120
#
# step = 0.01
# nfw_normalizations = np.arange(0.0, 0.8 + step, step)
#
# disk_min, disk_max, step = 0.25, 1.15, 0.01
# disk_normalization = np.arange(disk_min, disk_max+step, step)
#
# scale_h_step = 0.0025
# disk_scale_height = np.arange(0.23, 0.33 + scale_h_step, scale_h_step)
#
# np.savetxt(os.getenv('SCRATCH')+'/saved_potentials/nfw_norms.txt', nfw_normalizations)
# np.savetxt(os.getenv('SCRATCH')+'/saved_potentials/disk_norms.txt', disk_normalization)
# np.savetxt(os.getenv('SCRATCH')+'/saved_potentials/scale_heights.txt', disk_scale_height)
#
# counter = 1
# print('ntotal ', len(disk_normalization)*len(nfw_normalizations)*len(disk_scale_height))
# compute = True
# potential_list = []
#
# for nfw_norm in nfw_normalizations:
#     for disk_norm in disk_normalization:
#         for scale_height in disk_scale_height:
#             if counter == int(sys.argv[1]):
#                 if os.path.exists(os.getenv('SCRATCH') +'/saved_potentials/MW_' + str(counter-1)):
#                     continue
#
#                 galactic_potential = [PowerSphericalPotentialwCutoff(normalize=0.05, alpha=1.8,
#                                                                      rc=1.9 / 8.),
#                                       MiyamotoNagaiPotential(a=3. / 8., b=scale_height / 8., normalize=disk_norm),
#                                       NFWPotential(a=2., normalize=nfw_norm)]
#
#                 potential_local = PotentialExtension(galactic_potential, z_min_max_kpc, vz_min_max_kmsec, phase_space_N)
#
#                 f = open(os.getenv('SCRATCH') +'/saved_potentials/MW_' + str(counter-1), 'wb')
#                 pickle.dump(potential_local, f)
#                 f.close()
#         counter += 1
#
# print(len(potential_list))