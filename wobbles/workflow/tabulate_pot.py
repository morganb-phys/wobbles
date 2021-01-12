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

