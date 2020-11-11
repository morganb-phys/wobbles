import pytest
import numpy.testing as npt
import numpy as np
from galpy import util
import pickle
from wobbles.distribution_function import DistributionFunction, _SingleDistributionFunction

class TestDistributionFunction(object):

    def setup(self):

        f = open('./MW14pot_100', "rb")
        self.potential_extension_local = pickle.load(f)
        f.close()

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

        df = _SingleDistributionFunction(rho_midplane, vdis, J, nu, self.v_domain, self.z_domain, self.length_scale,
                 self.velocity_scale, self.density_scale)

        max_rho = max(df.density)
        npt.assert_almost_equal(max_rho, rho_midplane_physical, decimal=3)

        component_amplitude = [0.6, 0.4]
        rho_midplane_value = 0.1
        rho_midplane_physical = [rho_midplane_value * component_amplitude[0], rho_midplane_value * component_amplitude[1]]
        rho_midplane = [rho_midplane_physical[0]/self.density_scale, rho_midplane_physical[1]/self.density_scale]
        vdis = [20.5 / self.velocity_scale] * 2

        J = self.potential_extension_local.action
        nu = self.potential_extension_local.vertical_freq

        df1 = _SingleDistributionFunction(rho_midplane[0], vdis[0], J, nu, self.v_domain, self.z_domain, self.length_scale,
                                         self.velocity_scale, self.density_scale)
        df2 = _SingleDistributionFunction(rho_midplane[1], vdis[1], J, nu, self.v_domain, self.z_domain, self.length_scale,
                                         self.velocity_scale, self.density_scale)
        df3 = DistributionFunction(np.sum(rho_midplane), component_amplitude, vdis, J, nu, self.v_domain, self.z_domain, self.length_scale,
                 self.velocity_scale, self.density_scale)

        max_rho1 = max(df1.density)
        max_rho2 = max(df2.density)

        max_rho1_combined = max(df3.dF_list[0].density)
        max_rho2_combined = max(df3.dF_list[1].density)
        max_rho_combined = max(df3.density)

        npt.assert_almost_equal(max_rho1/max_rho2, component_amplitude[0]/component_amplitude[1])
        npt.assert_almost_equal(max_rho1, max_rho1_combined)
        npt.assert_almost_equal(max_rho2, max_rho2_combined)
        npt.assert_almost_equal(max_rho_combined, max_rho1 + max_rho2)


#
# t = TestDistributionFunction()
# t.setup()
# t.test_density()
# t.test_velocity()

# comment this out if you uncomment the lines above
if __name__ == '__main__':
   pytest.main()
