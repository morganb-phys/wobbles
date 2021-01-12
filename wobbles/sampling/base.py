import numpy as np
from copy import deepcopy
from wobbles.workflow.forward_model import single_iteration
from wobbles.sampling.prior_manager import UniformPrior, GaussianPrior

class Base(object):

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

        samples_prior_list = self.set_params(parameters_sampled)

        samples = self.prior_class.draw(samples_prior_list)

        asymmetry, mean_vz, density = single_iteration(samples, *self._args_sampler)

        return asymmetry, mean_vz, density

    def set_prior(self, prior_class):

        self._prior_set = True
        self.priors_under_hood, self.priors_over_hood, self.to_sample_list = \
            prior_class.split_under_over_hood
        self.prior_class = prior_class
        self._dim = len(self.priors_over_hood)

    def prior_loglike(self, parameters_sampled):

        prior_functions = self.prior_functions

        assert len(prior_functions) == len(parameters_sampled)

        log_like = 0.

        for (param, func) in zip(parameters_sampled, prior_functions):

            log_like += func.loglike(param)

        return log_like

    @property
    def prior_functions(self):

        if not hasattr(self, '_prior_functions'):
            _, priors_over_hood, _ = self.prior_class.split_under_over_hood
            func_list = []

            for param_prior in priors_over_hood:

                prior_type = param_prior[1]
                prior_args = param_prior[2]

                if prior_type == 'u':

                    func = UniformPrior(prior_args[0], prior_args[1])

                elif prior_type == 'g':

                    func = GaussianPrior(prior_args[0], prior_args[1])

                else:
                    raise Exception('prior type ' + str(prior_type) + ' not recognized')

                func_list.append(func)

            self._prior_functions = func_list

        return self._prior_functions

    def set_params(self, params_sampled):

        new_params = deepcopy(self.priors_under_hood)

        for j, param_name in enumerate(self.to_sample_list):
            new = [[param_name, 'f', params_sampled[j], False]]
            new_params += new
        return new_params