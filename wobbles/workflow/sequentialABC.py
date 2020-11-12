from wobbles.workflow.forward_model_subs import single_iteration
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
import astroabc

class Data(object):

    def __init__(self, z_observed, data, errors):
        assert len(z_observed) == len(data)
        assert len(errors) == len(data)
        self.zobs = z_observed
        self.data = data
        self.errors = errors

    def penalty(self, z_model, model, error_inflate=1.):
        return np.sqrt(self.chi_square(z_model, model, error_inflate))

    def chi_square(self, z_model, model, error_inflate=1.):
        interp_model = interp1d(z_model, model)

        exponent = 0
        for (z_i, data_i, error_i) in zip(self.zobs, self.data, self.errors):
            delta = interp_model(z_i) - data_i
            dx = delta / (error_i * error_inflate)
            exponent += dx ** 2

        return exponent / len(z_model)


class JointData(object):

    def __init__(self, data_1, data_2, sigma_1=1., sigma_2=1.):

        self.data_1, self.data_2 = data_1, data_2

        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

    def penalty(self, z_model_1, z_model_2, model_1, model_2):

        if self.sigma_1 is None:
            pen1 = 0
        else:
            pen1 = self.data_1.penalty(z_model_1, model_1, self.sigma_1)
        if self.sigma_2 is None:
            pen2 = 0
        else:
            pen2 = self.data_2.penalty(z_model_2, model_2, self.sigma_2)
        return pen1 + pen2


class Simulation(object):

    def __init__(self, to_sample_list, param_priors, args_sampler):

        self.to_sample_list = to_sample_list
        self.param_prior = param_priors
        self.args_sampler = args_sampler

        self._z1 = np.linspace(0, 2, 100)
        self._z2 = np.linspace(-2, 2, 100)

    def _set_params(self, params_sampled):

        new_params = deepcopy(self.param_prior)
        for j, param_name in enumerate(self.to_sample_list):
            new_prior = [[param_name, 'f', params_sampled[j], False]]
            new_params += new_prior
        return new_params

    @staticmethod
    def _array_to_dictionary(parameter_priors):

        samples = {}
        for param_prior in parameter_priors:

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
            samples[param_name] = value

        return samples

    def simulate(self, params_sampled):

        samples_prior_list = self._set_params(params_sampled)

        samples = self._array_to_dictionary(samples_prior_list)

        asymmetry, mean_vz = single_iteration(samples, *self.args_sampler)

        if asymmetry is None or mean_vz is None:
            asymmetry = 10000 * np.ones_like(self._z1)
            mean_vz = 10000 * np.ones_like(self._z2)

        return (asymmetry, mean_vz)

    def dfunc(self, joint_data_class, model):

        (asymmetry, mean_vz) = model
        rho = joint_data_class.penalty(self._z1, self._z2, asymmetry, mean_vz)
        return rho