import numpy as np
from galpy.potential import NFWPotential
from galpy.potential import MiyamotoNagaiPotential
from galpy.potential import PowerSphericalPotentialwCutoff
from galpy.potential import evaluateDensities
from scipy.optimize import minimize


def minimize_function(x, rho_nfw_target, rho_midplane_target, density_conversion):
    """
    Computes the chi^2 penalty for a galactic potential for the purpose of finding an NFWPotential and MiyamotoNagaiPotential
    circular velocity normalization that yeild the desired midplane and nfw physical densities
    :param x: numpy array of proposed disk and nfw circular velocity normalizations (in that order)
    :param rho_nfw_target: desired nfw_normalization in physical M_sun / pc^2
    :param rho_midplane_target: desired midplane density in physical M_sun / pc^2
    :param density_conversion: a conversion factor between galpy internal density units and physical M_sun / pc^2
    :return: chi^2 penalty
    """

    galactic_potential = [PowerSphericalPotentialwCutoff(normalize=0.05, alpha=1.8,
                                                         rc=1.9 / 8.),
                          MiyamotoNagaiPotential(a=3. / 8., b=0.28 / 8., normalize=x[0]),
                          NFWPotential(a=2., normalize=x[1])]
    nfw_potential = NFWPotential(a=2., normalize=x[1])

    rho = evaluateDensities(galactic_potential, R=1., z=0.) * density_conversion
    rho_nfw = evaluateDensities(nfw_potential, R=1, z=0.) * density_conversion
    dx = (rho - rho_midplane_target) ** 2 / 0.000001 ** 2 + (rho_nfw - rho_nfw_target) ** 2 / 0.000001 ** 2

    return dx ** 0.5

def solve_normalizations(rho_nfw_target, rho_midplane_target, density_conversion):

    """
    Solves for the circular velocity normalization parameters for MiyamotoNagai and NFWPotential circular velocities that
    yield the desired NFW and midplne densities
    :param rho_nfw_target: desired nfw_normalization in physical M_sun / pc^2
    :param rho_midplane_target: desired midplane density in physical M_sun / pc^2
    :param density_conversion: a conversion factor between galpy internal density units and physical M_sun / pc^2
    :return: MiyamotoNagai and NFWPotential circular velocity normalizations
    """

    nfw_norm_init = 0.35 * (rho_nfw_target / 0.007)
    rho_mid_init = 0.6 * (rho_midplane_target/0.1)
    disk_norm_init = rho_mid_init - nfw_norm_init

    x0 = np.array([disk_norm_init, nfw_norm_init])

    opt = minimize(minimize_function, x0=x0, args=(rho_nfw_target, rho_midplane_target, density_conversion),
                   method='Nelder-Mead')

    return opt['x'][0], opt['x'][1]

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

