import numpy as np
from scipy.integrate import simps
from wobbles.util import fit_sec_squared
from scipy.interpolate import interp1d

class DistributionFunction(object):

    def __init__(self, rho_midplane, normalization_list, velocity_dispersion_list, J, nu, v_domain, z_domain, length_scale,
                 velocity_scale, density_scale, fill_value_interp='extrapolate', interp_kind='cubic', z_ref=None):

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
        :param length_scale [PHYS]: a physical length scale for the vertical height
        :param velocity_scale [PHYS]: a physical velocity scale
        :param density_scale [PHYS]: a physical density scale
        """

        assert len(normalization_list) == len(velocity_dispersion_list)

        dF_list = []

        kwargs_interp = {'fill_value': fill_value_interp, 'kind': interp_kind}

        for norm, sigma in zip(normalization_list, velocity_dispersion_list):
            if sigma == 0:
                raise Exception('cannot specify a velocity dispersion == 0')
            f = _SingleDistributionFunction(norm * rho_midplane, sigma, J, nu, v_domain, z_domain, length_scale, velocity_scale,
                                            density_scale, z_ref=z_ref)
            dF_list.append(f)

            # this will either be computed from the first component of the model or fixed to the value of
            # z_sun given to the class
            z_ref = f.z_ref

        self._kwargs_interp = kwargs_interp
        self.dF_list = dF_list
        self.z_ref = z_ref
        self.z = self.dF_list[0].z
        self.v = self.dF_list[0].v
        self.weights = np.array(normalization_list) / np.sum(normalization_list)

    def velocity_moment(self, n):

        v_moment = 0
        for df, norm in zip(self.dF_list, self.weights):
            v_moment += norm * df.velocity_moment(n)
        return v_moment

    def density_at_z(self, z):

        rho_interp = self._interpolated_density
        return rho_interp(z)

    @property
    def A(self):

        log_density = np.log10(self.density)
        interp_log = interp1d(self.z, log_density,
                          **self._kwargs_interp)

        zmin_max = np.max(self.z)

        zplus = np.linspace(0, zmin_max, int(len(self.v)))
        zminus = np.linspace(0, -zmin_max, int(len(self.v)))

        log_rho_plus = interp_log(zplus)
        log_rho_minus = interp_log(zminus)

        rho_plus, rho_minus = 10 ** log_rho_plus, 10 ** log_rho_minus
        A = (rho_plus - rho_minus) / (rho_plus + rho_minus)
        return A

    @property
    def density(self):

        rho = 0
        # the density normalization is applied inside _SingleDistributionFunction
        for df in self.dF_list:
            rho += df.density
        return rho

    @property
    def mean_v(self):

        mean_v = 0
        for df, norm in zip(self.dF_list, self.weights):
            mean_v += df.mean_v * norm
        return mean_v

    @property
    def velocity_dispersion(self):

        return np.sqrt(self.velocity_moment(2) - self.velocity_moment(1) ** 2)

    @property
    def mean_v_relative(self):

        mean_v = self.mean_v
        return mean_v - np.mean(mean_v)

    @property
    def _interpolated_density(self):

        if not hasattr(self, '_rho_interp'):

            self._rho_interp = interp1d(self.z - self.z_ref, self.density)

        return self._rho_interp


class _SingleDistributionFunction(object):

    def __init__(self, rho_midplane, sigma, J, nu, v_domain, z_domain, length_scale,
                 velocity_scale, density_scale, z_ref=None):
        """ exp(J * nu / sigma^2)"""
        self.rho0 = rho_midplane
        self.sigma0 = sigma
        self.J = J
        self.nu = nu

        self._vdom = v_domain
        self._zdom = z_domain

        self.density_scale = density_scale
        self.velocity_scale = velocity_scale
        self.length_scale = length_scale

        self.z, self.v = z_domain * length_scale, v_domain * velocity_scale

        self._norm_density = 1.
        rho = self.density
        idx = int(len(rho)/2)
        self._norm_density = self.density_scale * rho_midplane / rho[idx]

        if z_ref is None:
            x = fit_sec_squared(self.density, self.z)
            self.z_ref = x[1]
        else:
            self.z_ref = z_ref

    @property
    def normalization(self):

        f0 = self.f0
        return np.sum(f0, axis=1)

    def velocity_moment(self, n):

        f0 = self.f0
        v_integrand = (self._vdom[None, :] * self.velocity_scale) ** n

        return simps(f0 * v_integrand, axis=1) / self.normalization

    @property
    def f0(self):

        exponent = -self.J * self.nu / self.sigma0 ** 2
        f0 = np.exp(exponent) / np.sqrt(2*np.pi)

        return f0 * self.rho0/self.sigma0

    @property
    def density(self):

        f0 = self.f0

        rho = simps(f0, self._vdom, axis=1) * self.density_scale

        return rho * self._norm_density

    @property
    def mean_v(self):

        return self.velocity_moment(1)

    @property
    def velocity_dispersion(self):

        return np.sqrt(self.velocity_moment(2) - self.velocity_moment(1) ** 2)
