from abcpy.continuousmodels import Uniform, Normal
import numpy as np


class PriorBase(object):

    def __init__(self, param_prior):

        """
        Format ['param_name', 'prior_type', [arg1, arg2], positive_definite_bool, under_hood_bool]

        """

        self.param_prior = param_prior

    @classmethod
    def from_params(cls, nfw_norm, disk_norm, log_sag_mass_DM, sag_mass2light, f_sub, log_mslope,
                 m_host, velocity_dispersion_1, component_amplitude_1, velocity_dispersion_2, component_amplitude_2,
                 velocity_dispersion_3, component_amplitude_3, orbit_ra, orbit_dec, orbit_z, orbit_pm_ra, orbit_pm_dec, orbit_los,
                 gal_norm):

        """
                Format ['param_name', 'prior_type', [arg1, arg2], positive_definite_bool, under_hood_bool]

                """
        param_prior = []
        param_prior += nfw_norm
        param_prior += disk_norm
        param_prior += log_sag_mass_DM
        param_prior += sag_mass2light
        param_prior += f_sub
        param_prior += log_mslope
        param_prior += m_host
        param_prior += velocity_dispersion_1
        param_prior += component_amplitude_1
        param_prior += velocity_dispersion_2
        param_prior += component_amplitude_2
        param_prior += velocity_dispersion_3
        param_prior += component_amplitude_3
        param_prior += orbit_ra
        param_prior += orbit_dec
        param_prior += orbit_z
        param_prior += orbit_pm_ra
        param_prior += orbit_pm_dec
        param_prior += orbit_los
        param_prior += gal_norm
        return PriorBase(param_prior)

    @staticmethod
    def draw(param_prior):

        param_name = param_prior[0]
        prior_type = param_prior[1]
        prior_args = param_prior[2]
        positive_definite = param_prior[3]
        if prior_type == 'g':
            value = np.random.normal(*prior_args)
        elif prior_type == 'u':
            value = np.random.uniform(*prior_args)
        elif prior_type == 'f':
            value = prior_args
        else:
            raise Exception('param prior ' + str(param_prior[0]) + ' not valid.')
        if positive_definite:
            value = abs(value)

        return param_name, value

    @property
    def split_under_over_hood(self):

        if not hasattr(self, '_priors_over_hood'):
            priors_under_hood, priors_over_hood = [], []
            param_names_sampled_over_hood = []

            for prior in self.param_prior:
                under_over_hood_bool = prior[-1]

                param_name = prior[0]

                if under_over_hood_bool is True:
                    priors_under_hood.append(prior[0:-1])
                elif under_over_hood_bool is False:
                    priors_over_hood.append(prior[0:-1])
                    param_names_sampled_over_hood.append(param_name)
                else:
                    raise Exception('parameter must be bool')

            self._priors_under_hood = priors_under_hood
            self._priors_over_hood = priors_over_hood
            self._param_names_sampled_over_hood = param_names_sampled_over_hood

        return self._priors_under_hood, self._priors_over_hood, self._param_names_sampled_over_hood

class UniformPrior(object):

    def __init__(self, low, high):

        self.low, self.high = low, high

    def loglike(self, param):

        if param < self.low:
            return -np.inf
        elif param > self.high:
            return -np.inf
        else:
            return 0.

class GaussianPrior(object):

    def __init__(self, mean, sigma):
        assert sigma > 0
        self.mean, self.sigma = mean, sigma

    def loglike(self, param):
        return -0.5 * (param - self.mean) ** 2 / self.sigma ** 2