from wobbles.distribution_function import DistributionFunction
from galpy.potential import evaluatezforces
from galpy import util
import numpy as np
from galpy.orbit import Orbit
from scipy.integrate import simps

from galpy.util.bovy_conversion import get_physical

class Disc(object):

    def __init__(self, potential_extension_local, potential_extension_global=None):

        """

        :param potential_extension_local: An instance of PotentialExtension used to compute properties of the local matter density
        :param potential_extension_global: An instance of PotentialExtension used to compute large scale properties of galaxy,
        for example the orbit of a perturber and the the suns position relative to the center of the galaxy
        """

        self.potential_extension_local = potential_extension_local
        if potential_extension_global is None:
            potential_extension_global = potential_extension_local
        self.potential_extension_global = potential_extension_global

        self._z_units_internal = potential_extension_local.z_units_internal
        self._v_units_internal = potential_extension_local.v_units_internal
        self._units = potential_extension_local.units

    def distribution_function(self, delta_action, velocity_dispersion_local, rho_midplane=None,
                              component_amplitude=None, component_densities=None, verbose=False,
                              kwargs_distribution_function={}):

        """
        This routine computes a distribution function for the vertical density and velocity around the sun given a
        perturbation to the action

        :param delta_action: a perturbation to the action
        :param velocity_dispersion_local: the local velocity dispersion of the disk
        :param rho_midplane: the midplane density of the disk. If not specified, it will be computed from the local_potential.
        For Isothermal potentials, you need to manually specify this as galpy will not compute it for you
        :param component_amplitude: the amplitude of each component of the distribution function, must sum to one. If
        not specified, a single component distribution function is assumed. If specified, must be a list the same length as
        velocity_dispersion_local.
        :param component_densities: overrides component_amplitude if specified (see documentation in compute_df)

        :return: An instance of DistributionFucntion (see wobbles.distribution_function)
        """

        density_scale = util.bovy_conversion.dens_in_msolpc3(self._units['vo'], self._units['ro'])
        velocity_scale = self._units['vo']
        length_scale = self._units['ro']

        if component_densities is not None:

            if not isinstance(component_densities, list):
                raise Exception('component densities should be a list with len(velocity_dispersion) '
                                'of densities in physical units M/pc^3')
            else:
                assert len(component_densities) == len(velocity_dispersion_local), 'each component density must correspond ' \
                                                                                   'to a unique velocity dispersion'

            if rho_midplane is not None:
                error_msg = 'You specified rho_midplane as well as rho_midplane, but you much choose one or the other.' \
                            'if component_densities is specified rho_midplane = sum(component_densities).'
                raise Exception(error_msg)
            if component_amplitude is not None:
                error_msg = 'You specified component_amplitude as well as component_densities, but you much choose one or the other.'
                raise Exception(error_msg)

            # DistributionFunction class expects density in internal units
            rho_midplane_physical = np.sum(component_densities)
            rho_midplane = rho_midplane_physical / density_scale

            component_amplitude = [density/rho_midplane_physical for density in component_densities]
            assert_sum_to_unity = False
        else:
            assert_sum_to_unity = True

        if rho_midplane is None:
            rho_midplane = self.potential_extension_local.rho_midplane
            if verbose:
                print('computed a midplane density of '+str(rho_midplane * density_scale) +' [Msun/pc^3]')
        else:
            if verbose:
                print('using a specified midplane density of '+ str(rho_midplane * density_scale) +' [Msun/pc^3]')

        if component_amplitude is None:
            component_amplitude = [1]
        else:
            if not isinstance(component_amplitude, list):
                raise Exception('If specified, component amplitude must be a list')

        if assert_sum_to_unity and np.sum(component_amplitude) != 1:
            raise Exception('component amplitudes must sum to one')

        if not isinstance(velocity_dispersion_local, list):
            velocity_dispersion_local = [velocity_dispersion_local]

        if len(velocity_dispersion_local) != len(component_amplitude):
            raise Exception('if component amplitude or component_density is specified as a list, it must be the same length as '
                            'velocity_dispersion')

        vdis_local_units_internal = []
        for vdis in velocity_dispersion_local:
            vdis_local_units_internal.append(vdis / velocity_scale)

        vertical_freq = self.potential_extension_local.vertical_freq

        if verbose:
            for i, (vdis, norm) in enumerate(zip(vdis_local_units_internal, component_amplitude)):
                print('velocity dispersion '+str(i)+' (km/sec): ', vdis * velocity_scale)
                print('amplitude of component '+str(i) + ': ', norm)
                print('density ' + str(i) + ' (km/sec): ', norm * rho_midplane * density_scale)
            print('vertical frequency: ', vertical_freq)
            print('\n')

        J = self.potential_extension_local.action + delta_action
        dF = DistributionFunction(rho_midplane, component_amplitude, vdis_local_units_internal, J, vertical_freq,
                                  self._v_units_internal, self._z_units_internal, length_scale,
                                  velocity_scale, density_scale, **kwargs_distribution_function)

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

        delta_J = simps(v_z * force, dx=time_step) / self.potential_extension_local.angle

        return delta_J

    def _satellite_force(self, sat_time, orb_time, satellite_orbit_physical_off, phase_space_orbits_physical_off,
                        satellite_potential_physical_off, verbose):

        r_over_r0 = self.potential_extension_global.R_over_R0_eval

        vc_over_v0 = self.potential_extension_global.Vc

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

        pot = self.potential_extension_local.vertical_disk_potential_physical_off
        orbits.integrate(time_units_internal, pot)

        self._orbits = orbits

        return self._orbits


