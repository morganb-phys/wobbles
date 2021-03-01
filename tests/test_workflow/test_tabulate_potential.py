import pytest
import numpy.testing as npt
import numpy as np
from scipy.optimize import minimize
from galpy.potential import NFWPotential, MiyamotoNagaiPotential, PowerSphericalPotentialwCutoff
from wobbles.potential_extension import PotentialExtension
from wobbles.workflow.tabulate_pot import TabulatedPotential, TabulatedPotential3D, solve_normalizations
from galpy.potential import evaluateDensities

class TestComputationRoutines(object):

    def setup(self):

        self.density_conversion = 0.1758345842
        self.rho_nfw_target = 0.0079 # in M_sun / pc^2
        self.rho_midplane_target = 0.15 # M_sun / pc^2

    def test_minimize_func(self):

        disk_norm, nfw_norm = solve_normalizations(self.rho_nfw_target, self.rho_midplane_target, self.density_conversion)

        pot = [PowerSphericalPotentialwCutoff(normalize=0.05, alpha=1.8,
                                                             rc=1.9 / 8.),
                              MiyamotoNagaiPotential(a=3. / 8., b=0.28 / 8., normalize=disk_norm),
                              NFWPotential(a=2., normalize=nfw_norm)]
        rho = evaluateDensities(pot, R=1., z=0) * self.density_conversion
        npt.assert_almost_equal(rho, self.rho_midplane_target, 3)

        disk_norm, nfw_norm = solve_normalizations(self.rho_nfw_target, self.rho_nfw_target * 0.5,
                                                   self.density_conversion)
        npt.assert_equal(True, disk_norm < 0)

class TestTabulatePotential2D(object):

    def setup(self):


        nfw_norm = [0.2, 0.4, 0.6]
        disk_norm = [0.15, 0.3, 0.45]
        pot_list = []

        for normnfw in nfw_norm:
            for normdisk in disk_norm:
                pot = [PowerSphericalPotentialwCutoff(normalize=0.05,alpha=1.8,
                                                 rc=1.9/8.),
                      MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=normdisk),
                      NFWPotential(a=2.,normalize=normnfw)]
                pot_extension = PotentialExtension(pot, 2, 120, 10)
                pot_list.append(pot_extension)

        self.disk_norm = disk_norm
        self.nfw_norm = nfw_norm
        self.pot_list = pot_list

        self.tabulated_potential = TabulatedPotential(self.pot_list, self.nfw_norm, self.disk_norm)

    def test_tabulated_potential(self):

        nfwnorm = 0.2
        disk_norm = 0.15
        output = self.tabulated_potential.evaluate(nfwnorm, disk_norm)
        output_action = output.action
        output_angle = output.angle
        npt.assert_almost_equal(output_action, self.pot_list[0].action)
        npt.assert_almost_equal(output_angle, self.pot_list[0].angle)

        nfwnorm = 0.38
        disk_norm = 0.35
        output = self.tabulated_potential.evaluate(nfwnorm, disk_norm)
        output_action = output.action
        output_angle = output.angle
        npt.assert_almost_equal(output_action, self.pot_list[4].action)
        npt.assert_almost_equal(output_angle, self.pot_list[4].angle)

        npt.assert_raises(AssertionError, self.tabulated_potential.evaluate, 100, disk_norm)
        npt.assert_raises(AssertionError, self.tabulated_potential.evaluate, nfwnorm, -100)

class TestTabulatePotential3D(object):

    def setup(self):

        nfw_norm = [0.2, 0.4, 0.6]
        disk_norm = [0.15, 0.3, 0.45]
        scale_height = [0.25, 0.27, 0.29]
        pot_list = []

        for normnfw in nfw_norm:
            for normdisk in disk_norm:
                for scale_h in scale_height:
                    pot = [PowerSphericalPotentialwCutoff(normalize=0.05,alpha=1.8,
                                                     rc=1.9/8.),
                          MiyamotoNagaiPotential(a=3./8.,b=scale_h/8.,normalize=normdisk),
                          NFWPotential(a=2.,normalize=normnfw)]
                    pot_extension = PotentialExtension(pot, 2, 120, 3, compute_action_angle=True)
                    pot_list.append(pot_extension)

        self.disk_norm = disk_norm
        self.nfw_norm = nfw_norm
        self.scale_height = scale_height
        self.pot_list = pot_list

        self.tabulated_potential = TabulatedPotential3D(self.pot_list, self.nfw_norm, self.disk_norm, self.scale_height)


    def test_tabulated_potential(self):

        nfwnorm = 0.2
        disk_norm = 0.15
        scale_height = 0.25
        output = self.tabulated_potential.evaluate(nfwnorm, disk_norm, scale_height)
        output_action = output.action
        output_angle = output.angle
        npt.assert_almost_equal(output_action, self.pot_list[0].action)
        npt.assert_almost_equal(output_angle, self.pot_list[0].angle)

        nfwnorm = 0.2
        disk_norm = 0.15
        scale_height = 0.287
        output = self.tabulated_potential.evaluate(nfwnorm, disk_norm, scale_height)
        output_action = output.action
        output_angle = output.angle
        npt.assert_almost_equal(output_action, self.pot_list[2].action)
        npt.assert_almost_equal(output_angle, self.pot_list[2].angle)

        nfwnorm = 0.2
        disk_norm = 0.29
        scale_height = 0.251
        output = self.tabulated_potential.evaluate(nfwnorm, disk_norm, scale_height)
        output_action = output.action
        output_angle = output.angle
        npt.assert_almost_equal(output_action, self.pot_list[3].action)
        npt.assert_almost_equal(output_angle, self.pot_list[3].angle)

        nfwnorm = 0.4
        disk_norm = 0.19
        scale_height = 0.287
        output = self.tabulated_potential.evaluate(nfwnorm, disk_norm, scale_height)
        output_action = output.action
        output_angle = output.angle
        npt.assert_almost_equal(output_action, self.pot_list[11].action)
        npt.assert_almost_equal(output_angle, self.pot_list[11].angle)

        nfwnorm = 0.58
        disk_norm = 0.44
        scale_height = 0.287
        output = self.tabulated_potential.evaluate(nfwnorm, disk_norm, scale_height)
        output_action = output.action
        output_angle = output.angle
        npt.assert_almost_equal(output_action, self.pot_list[-1].action)
        npt.assert_almost_equal(output_angle, self.pot_list[-1].angle)

        npt.assert_raises(AssertionError, self.tabulated_potential.evaluate, 100, disk_norm, 0.3)
        npt.assert_raises(AssertionError, self.tabulated_potential.evaluate, nfwnorm, -100, 0.3)
        npt.assert_raises(AssertionError, self.tabulated_potential.evaluate, nfwnorm, disk_norm, -100)

if __name__ == '__main__':
  pytest.main()
