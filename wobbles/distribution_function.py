import numpy as np
from scipy.integrate import simps
from wobbles.util import fit_sec_squared
from scipy.interpolate import interp1d

class DistributionFunction(object):

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
        z_len= len(self.z)
        
        zplus = np.linspace(0, zmin_max, z_len)
        zminus = np.linspace(0, -zmin_max, z_len)
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

    @property
    def dphi_dz(self):

        norm = self.normalization
        velocity_dispersion = self.velocity_dispersion

        dz = abs(self.z[1] - self.z[0])
        d_phi_dz = -np.gradient(velocity_dispersion, dz) / norm

        return d_phi_dz
