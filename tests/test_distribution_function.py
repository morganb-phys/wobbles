import pytest
import numpy.testing as npt
import numpy as np
from galpy import util
import pickle
from wobbles.distribution_function import DistributionFunction, _SingleDistributionFunction
import os

class TestDistributionFunction(object):

    def setup(self):

        path = os.getcwd()
        f = open(path + '/tests/MW14pot_100', "rb")
        self.potential_extension_global = pickle.load(f)
        f.close()
        self.potential_extension_local = self.potential_extension_global

        self.J = self.potential_extension_local.action
        self.nu = self.potential_extension_local.vertical_freq

        self.v_domain, self.z_domain = self.potential_extension_local.z_units_internal, \
                                       self.potential_extension_local.v_units_internal
        units = self.potential_extension_local.units
        self.length_scale = units['ro']
        self.velocity_scale = units['vo']
        self.density_scale = util.bovy_conversion.dens_in_msolpc3(units['vo'], units['ro'])

    def test_density(self):

        rho_midplane_physical = 0.1
        rho_midplane = rho_midplane_physical / self.density_scale
        vdis = 20.5 / self.velocity_scale
        J = self.potential_extension_local.action
        nu = self.potential_extension_local.vertical_freq

        kwargs_interp = {'fill_value': 'extrapolate', 'kind': 'linear'}
        df = _SingleDistributionFunction(rho_midplane, vdis, J, nu, self.v_domain, self.z_domain, self.length_scale,
                 self.velocity_scale, self.density_scale, kwargs_interp)

        max_rho = max(df.density)
        npt.assert_almost_equal(max_rho, rho_midplane_physical, decimal=3)

        component_amplitude = [0.6, 0.4]
        rho_midplane_value = 0.1
        rho_midplane_physical = [rho_midplane_value * component_amplitude[0], rho_midplane_value * component_amplitude[1]]
        rho_midplane = [rho_midplane_physical[0]/self.density_scale, rho_midplane_physical[1]/self.density_scale]
        vdis = [20.5 / self.velocity_scale] * 2

        J = self.potential_extension_local.action
        nu = self.potential_extension_local.vertical_freq

        kwargs_interp = {'fill_value': 'extrapolate', 'kind': 'cubic'}
        df1 = _SingleDistributionFunction(rho_midplane[0], vdis[0], J, nu, self.v_domain, self.z_domain, self.length_scale,
                                         self.velocity_scale, self.density_scale, kwargs_interp)
        kwargs_interp = {'fill_value': 'extrapolate', 'kind': 'cubic'}
        df2 = _SingleDistributionFunction(rho_midplane[1], vdis[1], J, nu, self.v_domain, self.z_domain, self.length_scale,
                                         self.velocity_scale, self.density_scale, kwargs_interp)
        df3 = DistributionFunction(np.sum(rho_midplane), component_amplitude, vdis, J, nu, self.v_domain, self.z_domain, self.length_scale,
                 self.velocity_scale, self.density_scale, fill_value_interp='extrapolate', interp_kind='cubic')

        max_rho1 = max(df1.density)
        max_rho2 = max(df2.density)

        max_rho1_combined = max(df3.dF_list[0].density)
        max_rho2_combined = max(df3.dF_list[1].density)
        max_rho_combined = max(df3.density)

        npt.assert_almost_equal(max_rho1/max_rho2, component_amplitude[0]/component_amplitude[1])
        npt.assert_almost_equal(max_rho1, max_rho1_combined)
        npt.assert_almost_equal(max_rho2, max_rho2_combined)
        npt.assert_almost_equal(max_rho_combined, max_rho1 + max_rho2)

    def test_with_perturbation(self):

        J = self.potential_extension_local.action
        nu = self.potential_extension_local.vertical_freq

        component_amplitude = [0.6, 0.4]
        rho_midplane_value = 0.1
        rho_midplane_physical = [rho_midplane_value * component_amplitude[0],
                                 rho_midplane_value * component_amplitude[1]]
        rho_midplane = [rho_midplane_physical[0] / self.density_scale, rho_midplane_physical[1] / self.density_scale]
        vdis = [20.5 / self.velocity_scale, 4 / self.velocity_scale]

        delta_J = np.loadtxt(os.getcwd() + '/tests/delta_J_test.txt')

        df3 = DistributionFunction(np.sum(rho_midplane), component_amplitude, vdis, J + delta_J, nu,
                                         self.v_domain, self.z_domain,
                                         self.length_scale,
                                         self.velocity_scale, self.density_scale, fill_value_interp='extrapolate',
                                         interp_kind='linear')

        npt.assert_array_less(np.absolute(df3.A), 1)

    def test_function(self):

        rho_midplane = 0.1
        component_amplitude = [0.6, 0.3]
        vdis = [20.5 / self.velocity_scale, 10 / self.velocity_scale]
        J = self.potential_extension_local.action
        delta_J = np.loadtxt(os.getcwd() + '/tests/delta_J_test.txt')
        nu = self.potential_extension_local.vertical_freq

        df = DistributionFunction(rho_midplane, component_amplitude, vdis, J + delta_J, nu,
                                   self.v_domain, self.z_domain,
                                   self.length_scale,
                                   self.velocity_scale, self.density_scale, fill_value_interp='extrapolate',
                                   interp_kind='linear')

        function = df.function
        weights = np.array(component_amplitude) / np.sum(component_amplitude)
        function_2 = df.dF_list[0].f0 * weights[0] + df.dF_list[1].f0 * weights[1]
        npt.assert_almost_equal(function, function_2)



if __name__ == '__main__':
   pytest.main()
