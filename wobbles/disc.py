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

    def distribution_function(self, delta_action, rho_midplane=None):

        """
        This routine computes a distribution function for the vertical density and velocity around the sun given a
        perturbation to the action

        :param delta_action: a perturbation to the action
        :param rho_midplane: the midplane density of the disk. If not specified, it will be computed from the local_potential.
        For Isothermal potentials, you need to manually specify this as galpy will not compute it for you

        :return: An instancce of DistributionFucntion (see wobbles.distribution_function)
        """

        if rho_midplane is None:
            rho_midplane = self.potential_extension.rho_midplane
        velocity_dispersion_local = self.potential_extension.velocity_dispersion_local
        vertical_freq = self.potential_extension.vertical_freq

        density_scale = util.bovy_conversion.dens_in_msolpc3(self._units['vo'], self._units['ro'])
        velocity_scale = self._units['vo']
        length_scale = self._units['ro']

        dF = DistributionFunction(rho_midplane, velocity_dispersion_local,
                                  self.potential_extension.action + delta_action,
                                  vertical_freq, self._v_units_internal,
                                  self._z_units_internal, length_scale, velocity_scale, density_scale)

        return dF

    def satellite_forces(self, time_units_internal, satellite_orbit_list, satellite_potentials_list):

        """
        Computes the force exterted by a passing satellite (or satellites) in the z direction

        :param time_units_internal: the time over which to compute the perturbation specified in galpy internal units
        Should be computed from the time over which the satellite perturbtation is computed, but is not necessarily the same
        :param satellite_orbit_list: a list of perturbing satellite orbits (instances of galpy.orbit)
        :param satellite_potentials_list: a list of perturbing satellite potentials; should be the same length as satellite_orbit_list

        :return: the force from the passing satellite at each point in the z direction
        """

        assert len(satellite_orbit_list) == len(satellite_potentials_list)
        phase_space_orbits = self.orbits_in_phase_space(time_units_internal)
        force = 0
        for (orbit, potential) in zip(satellite_orbit_list, satellite_potentials_list):
            new_force = self._satellite_force(time_units_internal, orbit, phase_space_orbits, potential)
            force += new_force

        return force

    def action_impulse(self, force, time_internal_units, satellite_orbit_list, satellite_potentials_list):

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

        phase_space_orbits = self.orbits_in_phase_space(time_internal_units)
        v_z = phase_space_orbits.vx(time_internal_units)

        time_step = time_internal_units[1] - time_internal_units[0]

        delta_J = simps(v_z * force) * time_step / self.potential_extension.angle

        return delta_J

    def _satellite_force(self, time, satellite_orbit_physical_off, phase_space_orbits_physical_off,
                        satellite_potential_physical_off):

        r_over_r0 = self.potential_extension.R_over_R0_eval

        vc_over_v0 = self.potential_extension.Vc

        freq = vc_over_v0 / r_over_r0

        dx = r_over_r0 * np.cos(freq * time) - satellite_orbit_physical_off.x(time)
        dy = r_over_r0 * np.sin(freq * time) - satellite_orbit_physical_off.y(time)
        dz = phase_space_orbits_physical_off.x(time) - satellite_orbit_physical_off.z(time)
        dR = np.sqrt(dx ** 2. + dy ** 2.)

        force = evaluatezforces(satellite_potential_physical_off, R=dR, z=dz)

        return force

    def orbits_in_phase_space(self, time_units_internal):

        # only need to compute this once
        if not hasattr(self, '_orbits'):

            vxvv = np.array(np.meshgrid(self._z_units_internal, self._v_units_internal)).T

            # the units of ro and vo we need for the orbits in phase space are that of the local potential, not
            # the ones used to compute the satellite orbit
            orbits = Orbit(vxvv, ro=self._units['ro'], vo=self._units['vo'])

            orbits.turn_physical_off()

            pot = self.potential_extension.vertical_disk_potential_physical_off
            orbits.integrate(time_units_internal, pot)
            self._orbits = orbits

        return self._orbits


