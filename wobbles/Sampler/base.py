import numpy as np
from copy import deepcopy
from wobbles.workflow.forward_model import single_iteration

class MCMCBase(object):

    def __init__(self, output_folder, args_sampler,
                 observed_data,
                 observed_data_z_eval):

        # (self._tabpot, self._kde, self._phase_space_res) = args_sampler
        self._args_sampler = args_sampler

        self.output_folder = output_folder

        self._data_domain = observed_data_z_eval
        self._observed_data = observed_data

        self._prior_set = False

    def model_data_from_params(self, parameters_sampled):

        samples_prior_list = self._set_params(parameters_sampled)

        samples = {}
        for param_prior in samples_prior_list:
            param_name, value = self.prior_class.draw(param_prior)
            samples[param_name] = value

        asymmetry, mean_vz = single_iteration(samples, *self._args_sampler)

        return asymmetry, mean_vz

    def set_prior(self, prior_class):

        self._prior_set = True
        self.priors_under_hood, self.priors_over_hood, self.to_sample_list = \
            prior_class.split_under_over_hood
        self.prior_class = prior_class
        self._dim = len(self.priors_over_hood)

    def log_prior(self, parameters_sampled):

        bounds = self.bounds

        weight = 0.

        for i, param in enumerate(parameters_sampled):

            if param < bounds[0][i] or param > bounds[1][i]:
                return np.nan
            else:

                weight += bounds[2][i](param)

        return weight

    @property
    def bounds(self):

        if not hasattr(self, '_bounds'):
            _, priors_over_hood, _ = self.prior_class.split_under_over_hood
            bound_low, bound_high = [], []
            log_weights = []
            for param_prior in priors_over_hood:

                prior_type = param_prior[1]
                prior_args = param_prior[2]
                if prior_type == 'u':
                    bound_low.append(prior_args[0])
                    bound_high.append(prior_args[1])
                    w = lambda x: 0.
                    log_weights.append(w)
                elif prior_type == 'g':
                    bound_low.append(-np.inf)
                    bound_high.append(np.inf)
                    w = lambda x: -0.5 * (x - prior_args[0]) ** 2 / prior_args[1] ** 2
                    log_weights.append(w)
                else:
                    raise Exception('prior type ' + str(prior_type) + ' not recognized')

            bound_low, bound_high = np.array(bound_low), np.array(bound_high)
            self._bounds = (bound_low, bound_high, log_weights)

        return self._bounds

    def _set_params(self, params_sampled):

        new_params = deepcopy(self.priors_under_hood)

        for j, param_name in enumerate(self.to_sample_list):
            new = [[param_name, 'f', params_sampled[j], False]]
            new_params += new
        return new_params