import numpy as np
from scipy.integrate import simps
from wobbles.util import fit_sec_squared
from scipy.interpolate import interp1d

class DistributionFunction(object):

    def __init__(self, rho_midplane, normalization_list, velocity_dispersion_list, J, nu, v_domain, z_domain, length_scale,
                 velocity_scale, density_scale):

        """
        Constructs a distribution function for the disk as a sum of quasi-isothermal distribution functions

        unit: [GIU] means [galpy internal units]
        unit: [PHYS] means a physical unit
        unit: [F] means a floating point number

        :param rho_midplane [GIU]: the midplane density of the disk in galpy internal units
        :param normalization_list [F]: a list of normalizations for each component; must add to one
        :param velocity_dispersion_list [GIU]: a list of velocity dispersions for each component of the disk
        :param J [GIU]: the vertical action computed for each point in phase space
        :param nu [GIU]: the vertical frequency of the disk
        :param v_domain [GIU]: the velocity domain over which the phase space distribution is computed
        :param z_domain [GIU]: the vertial height over which the phase space distribution is computed
        :param length_scale [PHYS]: a physial length scale for the vertical height
        :param velocity_scale [PHYS]: a physical velocity scale
        :param density_scale [PHYS]: a physical density scale
        """

        assert np.sum(normalization_list) == 1
        assert len(normalization_list) == len(velocity_dispersion_list)

        dF_list = []
        for norm, sigma in zip(normalization_list, velocity_dispersion_list):

            f = _SingleDistributionFunction(norm * rho_midplane, sigma, J, nu, v_domain, z_domain, length_scale, velocity_scale,
                                            density_scale)
            dF_list.append(f)

        self.dF_list = dF_list

    def velocity_moment(self, n):

        v_moment = 0
        for df in self.dF_list:
            v_moment += df.velocity_moment(n)
        return v_moment

    @property
    def A(self):

        A = 0
        for df in self.dF_list:
            A += df.A
        return A

    @property
    def density(self):

        rho = 0
        for df in self.dF_list:
            rho += df.density
        return rho

    @property
    def mean_v(self):

        mean_v = 0
        for df in self.dF_list:
            mean_v += df.mean_v
        return mean_v

    @property
    def velocity_dispersion(self):

        vdis = 0
        for df in self.dF_list:
            vdis += df.velocity_dispersion
        return vdis

    @property
    def mean_v_relative(self):

        mean_v_rel = 0
        for df in self.dF_list:
            mean_v_rel += df.mean_v_relative
        return mean_v_rel


class _SingleDistributionFunction(object):

    def __init__(self, rho, sigma, J, nu, v_domain, z_domain, length_scale,
                 velocity_scale, density_scale):
        """ exp(J * nu / sigma^2)"""
        self.rho0 = rho
        self.sigma0 = sigma
        self.J = J
        self.nu = nu

        self._vdom = v_domain
        self._zdom = z_domain

        self.density_scale = density_scale
        self.velocity_scale = velocity_scale
        self.length_scale = length_scale

        self.z, self.v = z_domain * length_scale, v_domain * velocity_scale

        x = fit_sec_squared(self.density, self.z)
        self.z_fit = x[1]

    @property
    def normalization(self):

        f0 = self.f0
        return np.sum(f0, axis=1)

    def velocity_moment(self, n):

        f0 = self.f0
        v_integrand = (self._vdom[None, :] * self.velocity_scale) ** n

        return simps(f0 * v_integrand, axis=1) / self.normalization

    @property
    def A(self):

        interp = interp1d(self.z + self.z_fit, self.density, fill_value='extrapolate', kind='cubic')

        zmin_max = np.max(self.z)

        zplus = np.linspace(0, zmin_max, 100)
        zminus = np.linspace(0, -zmin_max, 100)
        rho_plus = interp(zplus)
        rho_minus = interp(zminus)
        A = (rho_plus - rho_minus) / (rho_plus + rho_minus)
        return A

    @property
    def f0(self):

        exponent = -self.J * self.nu / self.sigma0 ** 2
        f0 = np.exp(exponent) / np.sqrt(2*np.pi)

        return f0 * self.rho0/self.sigma0

    @property
    def density(self):

        f0 = self.f0

        rho = simps(f0, self._vdom, axis=1) * self.density_scale

        return rho

    @property
    def mean_v(self):

        return self.velocity_moment(1)

    @property
    def velocity_dispersion(self):

        return np.sqrt(self.velocity_moment(2) - self.velocity_moment(1) ** 2)

    @property
    def mean_v_relative(self):

        v = self.mean_v
        return v - np.mean(v)
