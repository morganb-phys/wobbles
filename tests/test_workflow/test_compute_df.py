import pytest
import numpy.testing as npt
import pickle
import numpy as np
import astropy.units as apu
from wobbles.workflow.integrate_single_orbit import integrate_orbit
import galpy
from galpy.potential import HernquistPotential
from wobbles.disc import Disc
from wobbles.workflow.compute_distribution_function import compute_df
import os

class TestComputeDistributionFunction(object):

    def setup(self):

        t_orbit = -1.64  # Gyr
        N_tsteps = 600
        self.time_Gyr = np.linspace(0., t_orbit, N_tsteps) * apu.Gyr

        path_to_MWpot100 = os.getcwd() + '/tests/'
        f = open(path_to_MWpot100 + 'MW14pot_100', "rb")
        self.potential_extension_global = pickle.load(f)
        f.close()

        self.potential_extension_local = self.potential_extension_global

        orbit_init_sag = [283. * apu.deg, -30. * apu.deg, 26. * apu.kpc,
                          -2.6 * apu.mas / apu.yr, -1.3 * apu.mas / apu.yr,
                          140. * apu.km / apu.s]  # Initial conditions of the satellite
        sag_orbit_phsical_off = integrate_orbit(orbit_init_sag, self.potential_extension_global.galactic_potential, self.time_Gyr)
        self.sag_orbit = [sag_orbit_phsical_off]

        sag_potential_1 = galpy.potential.HernquistPotential(amp=1. * 1e10 * apu.M_sun, a=3. * apu.kpc)
        sag_potential_2 = galpy.potential.HernquistPotential(amp=1. * 0.2e9 * apu.M_sun, a=0.65 * apu.kpc)
        self.sag_potential = [sag_potential_1 + sag_potential_2]
        galpy.potential.turn_physical_off(self.sag_potential)

        self.disc = Disc(self.potential_extension_local, self.potential_extension_global)

        self.time_internal_units = self.sag_orbit[0].time()

        velocity_dispersion_local = [20.5, 20.5]
        normalizations = [0.7, 0.3]
        self.dF1, self.delta_J1, self.force1 = compute_df(self.disc, self.time_internal_units,
                                           self.sag_orbit, self.sag_potential,
                                           velocity_dispersion_local, normalizations,
                                           verbose=True)

        velocity_dispersion_local = [20.5]
        normalizations = [1.]
        self.dF2, self.delta_J2, self.force2 = compute_df(self.disc, self.time_internal_units,
                                           self.sag_orbit, self.sag_potential,
                                           velocity_dispersion_local, normalizations,
                                           verbose=True)

        self.component_densities = [0.7 * 0.1, 0.3 * 0.1]
        velocity_dispersion_local = [20.5, 20.5]
        self.dF3, self.delta_J3, self.force3 = compute_df(self.disc, self.time_internal_units,
                                                          self.sag_orbit, self.sag_potential,
                                                          velocity_dispersion_local,
                                                          component_densities=self.component_densities,
                                                          verbose=True)


        sag_potential_1 = galpy.potential.HernquistPotential(amp=0. * apu.M_sun, a=3. * apu.kpc)
        sag_potential_2 = galpy.potential.HernquistPotential(amp=0. * apu.M_sun, a=0.65 * apu.kpc)
        sag_potential_nopert = [sag_potential_1 + sag_potential_2]
        galpy.potential.turn_physical_off(sag_potential_nopert)
        velocity_dispersion_local = 20.5
        normalizations = [1.]
        self.dF_nopert, self.delta_J_nopert, self.force_nopert = compute_df(self.disc, self.time_internal_units,
                                                          self.sag_orbit, sag_potential_nopert,
                                                          velocity_dispersion_local, normalizations,
                                                          verbose=True)

    def test_multi_component(self):

        npt.assert_almost_equal(self.force1, self.force2)
        npt.assert_almost_equal(self.delta_J1, self.delta_J2)

        A1, A2 = self.dF1.A, self.dF2.A
        npt.assert_almost_equal(A1, A2, 5)

        vz1, vz2 = self.dF1.mean_v, self.dF2.mean_v
        npt.assert_almost_equal(vz1, vz2, 5)
        vz1, vz2 = self.dF1.mean_v_relative, self.dF2.mean_v_relative
        npt.assert_almost_equal(vz1, vz2, 5)

        vdis1, vdis2 = self.dF1.velocity_dispersion, self.dF2.velocity_dispersion
        npt.assert_almost_equal(vdis1, vdis2)

    def test_multi_component_density(self):

        rho2 = self.dF2.density
        rho3 = self.dF3.density

        ratio = max(rho3)/max(rho2)
        print(ratio)

    def test_no_perturbation(self):

        npt.assert_almost_equal(self.delta_J_nopert, np.zeros_like(self.delta_J_nopert))
        npt.assert_almost_equal(self.force_nopert, np.zeros_like(self.force_nopert))
        A = self.dF_nopert.A[1:-1]
        npt.assert_almost_equal(A, np.zeros_like(A), decimal=3)

        vz = self.dF_nopert.mean_v_relative[1:-1]
        npt.assert_almost_equal(vz, np.zeros_like(vz), decimal=3)

    def test_exceptions(self):

        velocity_dispersion_local = [20.5, 20.5]
        normalizations = [0.5, 0.3]
        args = (self.disc, self.time_internal_units, self.sag_orbit, self.sag_potential,
                                                          velocity_dispersion_local, normalizations)
        npt.assert_raises(Exception, compute_df, args)
#

if __name__ == '__main__':
    pytest.main()

