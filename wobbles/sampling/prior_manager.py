from scipy.stats import multivariate_normal
import numpy as np


class PriorBase(object):

    def __init__(self, param_prior):

        """
        Format ['param_name', 'prior_type', [arg1, arg2], positive_definite_bool, under_hood_bool]

        """

        self.param_prior = param_prior

    def draw(self, param_prior_list):

        samples = {}
        for param_prior in param_prior_list:
            _param_name, _param_value = self._draw_single(param_prior)

            for pname, value in zip(_param_name, _param_value):
                samples[pname] = value
        return samples

    @staticmethod
    def _draw_single(param_prior):

        param_name = param_prior[0]
        prior_type = param_prior[1]
        prior_args = param_prior[2]
        positive_definite = param_prior[3]

        if isinstance(param_name, list):
            assert prior_type == 'jg'
            pdf = multivariate_normal(*prior_args)
            values = pdf.rvs(1)
            return param_name, values

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

        return [param_name], [value]

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