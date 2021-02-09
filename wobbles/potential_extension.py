import galpy
from galpy.actionAngle.actionAngleVertical import actionAngleVertical
from galpy.potential import evaluatezforces, turn_physical_off, evaluatelinearPotentials
from galpy.util.bovy_conversion import get_physical, time_in_Gyr
import numpy as np

class PotentialExtension(object):

    valid_potential = True

    def __init__(self, galactic_potential, z_min_max_kpc, vz_min_max_kmsec, phase_space_N, R_over_R0_eval=1.,
                 velocity_dispersion_local=20.5, compute_action_angle=True):

        """
        This class handles some computations that are relevant to the problem of perturbations to the phase space
        in the solar neighborhood.

        :param galactic_potential: A 3D galpy potential, this is meant to model the local dynamics of stars, it is not necessarily
        the same orbit used to integrate a perturbing satellite
        :param z_min_max_kpc: the min/max vertical height [kpc]
        :param vz_min_max_kmsec: the min/max vertical velocity [km/sec]
        :param phase_space_N: the number of phase space coordinates, will compute phase_space_N^2 action angle variables in total
        :param R_over_R0_eval: The radius were the phase space is computed. Defaults to 1, i.e. the sun's
        position in the galaxy in internal units
        :param velocity_dispersion_local: the local velocity dispersion in km/sec

        """
        self.galactic_potential = galactic_potential

        self.R_over_R0_eval = R_over_R0_eval
        self._velocity_dispersion_local = velocity_dispersion_local
        self.vertical_disk_potential = [galpy.potential.toVerticalPotential(galactic_potential, self.R_over_R0_eval,
                                                                            phi=0.)]

        self.z_units_internal, self.v_units_internal, galactic_potential_physical_off, self.units = self._coordinate_system(z_min_max_kpc,
                                                                              vz_min_max_kmsec, phase_space_N, galactic_potential)

        self.velocity_scale = self.units['vo']
        self.length_scale = self.units['ro']

        self.galactic_potential_physical_off = galactic_potential_physical_off
        self.vertical_disk_potential_physical_off = [galpy.potential.toVerticalPotential(galactic_potential_physical_off,
                                                                                         self.R_over_R0_eval, phi=0.)]

        if compute_action_angle:
            self.action, self.angle = self.action_angle()
        else:
            self.action, self.angle = None, None

    def action_angle(self):

        """
        Computes the action angle coordiante at each point in phase space
        :return: action and angles
        """

        nz, nv = len(self.z_units_internal), len(self.v_units_internal)
        z_step = self.z_units_internal[1] - self.z_units_internal[0]
        v_step = self.v_units_internal[1] - self.v_units_internal[0]

        action = np.empty([nz, nv])
        angle = np.empty([nz, nv])
        tol = 1e-9
        for i, zval in enumerate(self.z_units_internal):

            for j, vval in enumerate(self.v_units_internal):

                if np.abs(zval) < tol:
                    zval = zval + z_step * 0.01
                if np.abs(vval) < tol:
                    vval = vval + v_step * 0.01

                aAV = actionAngleVertical(0.1, 0.1, 0.1, zval, vval, pot=self.vertical_disk_potential_physical_off)

                action[i, j] = aAV.Jz()

                angle[i, j] = 2. * np.pi / aAV.Tz()
                assert np.isfinite(action[i, j])
                assert np.isfinite(angle[i, j])

        return action, angle

    def time_to_internal_time(self, time):

        """

        :param time: time in Gyr
        :return: time expressed in internal time units
        """

        return time / time_in_Gyr(self.units['vo'], self.units['ro'])

    @property
    def Vc(self):

        """
        TODO: GENERALIZE THIS FOR ISOTHERMAL MODELS THAT DON'T HAVE A CIRCULAR VELOCITY
        :return: the circular velocity evaluated at R_over_R0 in internal units
        """
        try:
            Vc = galpy.potential.vcirc(self.galactic_potential, self.R_over_R0_eval)
        except:
            print('Potential has no attribute circular velocity, using 1. instead...')
            Vc = 1.
        return Vc

    @property
    def rho_midplane(self):

        """

        :return: the circular velocity evaluated at R_over_R0 in internal units.
        """

        rho_midplane = galpy.potential.evaluateDensities(self.galactic_potential, self.R_over_R0_eval, 0., phi=0.)

        return rho_midplane

    @property
    def vertical_freq(self):

        """

        :return: the second derivative of the local potential at z = 0
        """
        if not hasattr(self, '_d2psi_dz2'):
            vertical_disk_potential = self.vertical_disk_potential_physical_off
            z_eval = 0.
            npts = 10001
            zz = np.linspace(z_eval - 0.1, z_eval + 0.1, npts)
            d2psi_dz2_squared = np.gradient(np.gradient(evaluatelinearPotentials(vertical_disk_potential, zz), zz), zz)[
                int((npts - 1) / 2)]
            self._d2psi_dz2 = np.sqrt(d2psi_dz2_squared)

        return self._d2psi_dz2

    @property
    def velocity_dispersion_local(self):

        """

        :return: The local velocity dispersion
        """

        return self._velocity_dispersion_local / self.units['vo']

    @staticmethod
    def _coordinate_system(zmin_max, vmin_max, N, galactic_potential):

        """

        :param zmin_max:
        :param vmin_max:
        :param N:
        :param galactic_potential:
        :return:
        """
        _z = np.linspace(-zmin_max, zmin_max, N)
        _v = np.linspace(-vmin_max, vmin_max, N)

        turn_physical_off(galactic_potential)
        units = get_physical(galactic_potential)

        z_units_internal = _z / units['ro']
        v_units_internal = _v / units['vo']

        return z_units_internal, v_units_internal, galactic_potential, units

