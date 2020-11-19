import pytest
import numpy.testing as npt
import numpy as np
from wobbles.distribution_function import DistributionFunction, _SingleDistributionFunction
from galpy.potential import MWPotential2014, NFWPotential, MiyamotoNagaiPotential, PowerSphericalPotentialwCutoff
from wobbles.potential_extension import PotentialExtension
from wobbles.workflow.tabulate_pot import TabulatedPotential

class TestTabulatePotential(object):

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

if __name__ == '__main__':
  pytest.main()
