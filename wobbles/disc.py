from wobbles.distribution_function import DistributionFunction
from galpy.potential import evaluatezforces
from galpy import util
import numpy as np
from galpy.orbit import Orbit
from scipy.integrate import simps

from galpy.util.bovy_conversion import get_physical

class Disc(object):

    def __init__(self, potential_extension_local):

        self.potential_extension = potential_extension_local
        self._z_units_internal = potential_extension_local.z_units_internal
        self._v_units_internal = potential_extension_local.v_units_internal
        self._units = potential_extension_local.units

    def distribution_function(self, delta_action, rho_midplane=None, verbose=False):

        """
        This routine computes a distribution function for the vertical density and velocity around the sun given a
        perturbation to the action

        :param delta_action: a perturbation to the action
        :param rho_midplane: the midplane density of the disk. If not specified, it will be computed from the local_potential.
        For Isothermal potentials, you need to manually specify this as galpy will not compute it for you

        :return: An instance of DistributionFucntion (see wobbles.distribution_function)
        """

        density_scale = util.bovy_conversion.dens_in_msolpc3(self._units['vo'], self._units['ro'])
        velocity_scale = self._units['vo']
        length_scale = self._units['ro']

        if rho_midplane is None:
            rho_midplane = self.potential_extension.rho_midplane
            if verbose:
                print('computed a midplane density of '+str(rho_midplane * density_scale) +' [Msun/pc^3]')
        else:
            assert rho_midplane is not None
            if verbose:
                print('using a specified midplane density of '+ str(rho_midplane * density_scale) +' [Msun/pc^3]')

        velocity_dispersion_local = self.potential_extension.velocity_dispersion_local
        vertical_freq = self.potential_extension.vertical_freq

        if verbose:
            print('local velocity dispersion (km/sec): ', velocity_dispersion_local * velocity_scale)
            print('vertical frequency: ', vertical_freq)

        dF = DistributionFunction(rho_midplane, velocity_dispersion_local,
                                  self.potential_extension.action + delta_action,
                                  vertical_freq, self._v_units_internal,
                                  self._z_units_internal, length_scale, velocity_scale, density_scale)

        return dF

    def satellite_forces(self, t_eval_satellite, t_eval_orbits, satellite_orbit_list, satellite_potentials_list,
                         phase_space_orbits, verbose=False):

        """
        Computes the force exterted by a passing satellite (or satellites) in the z direction

        :param t_eval_satellite: the times at which to compute the perturbation from the satellite specified in galpy internal units
        :param t_eval_orbits: the times at which to evaluate the orbits in phase space
        :param satellite_orbit_list: a list of perturbing satellite orbits (instances of galpy.orbit)
        :param satellite_potentials_list: a list of perturbing satellite potentials; should be the same length as satellite_orbit_list

        :return: the force from the passing satellite at each point in the z direction
        """

        assert len(satellite_orbit_list) == len(satellite_potentials_list)

        force = 0

        for (orbit, potential) in zip(satellite_orbit_list, satellite_potentials_list):
            new_force = self._satellite_force(t_eval_satellite, t_eval_orbits, orbit,
                                              phase_space_orbits, potential, verbose)
            force += new_force

        return force

    def action_impulse(self, force, time_internal_units, satellite_orbit_list, satellite_potentials_list, phase_space_orbits):

        """
        Computes the perturbation to the action from the passing satellite

        :param force: the force from the passing satellite (see satellite_forces routine)
        :param time_units_internal: the time over which to compute the perturbation specified in galpy internal units
        Should be computed from the time over which the satellite perturbtation is computed, but is not necessarily the same
        :param satellite_orbit_list: a list of perturbing satellite orbits (instances of galpy.orbit)
        :param satellite_potentials_list: a list of perturbing satellite potentials; should be the same length as satellite_orbit_list

        :return: the change in the action caused by the passing satellite at each point in phase space
        shape: (len(self._z_units_internal), len(self._v_units_internal))
        """

        assert len(satellite_orbit_list) == len(satellite_potentials_list)

        v_z = phase_space_orbits.vx(time_internal_units)

        time_step = time_internal_units[1] - time_internal_units[0]

        delta_J = simps(v_z * force, dx=time_step) / self.potential_extension.angle

        return delta_J

    def _satellite_force(self, sat_time, orb_time, satellite_orbit_physical_off, phase_space_orbits_physical_off,
                        satellite_potential_physical_off, verbose):

        r_over_r0 = self.potential_extension.R_over_R0_eval

        vc_over_v0 = self.potential_extension.Vc

        freq = vc_over_v0 / r_over_r0

        if verbose:
            print('evaluating at r_ovver_r0 = '+str(r_over_r0))
            print('evaluating at vc_over_v0 = ' + str(vc_over_v0))

        dx = r_over_r0 * np.cos(freq * sat_time) - satellite_orbit_physical_off.x(sat_time)
        dy = r_over_r0 * np.sin(freq * sat_time) - satellite_orbit_physical_off.y(sat_time)
        dz = phase_space_orbits_physical_off.x(orb_time) - satellite_orbit_physical_off.z(sat_time)
        dR = np.sqrt(dx ** 2. + dy ** 2.)

        force = evaluatezforces(satellite_potential_physical_off, R=dR, z=dz)

        return force

    def orbits_in_phase_space(self, time_units_internal):

        vxvv = np.array(np.meshgrid(self._z_units_internal, self._v_units_internal)).T

        # the units of ro and vo we need for the orbits in phase space are that of the local potential, not
        # the ones used to compute the satellite orbit
        orbits = Orbit(vxvv, ro=self._units['ro'], vo=self._units['vo'])

        orbits.turn_physical_off()

        pot = self.potential_extension.vertical_disk_potential_physical_off
        orbits.integrate(time_units_internal, pot)

        self._orbits = orbits

        return self._orbits


