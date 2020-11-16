import pyswarms
import numpy as np
from copy import deepcopy
from wobbles.workflow.forward_model import single_iteration
from wobbles.Sampler.data import DistanceCalculator


class ParticleSwarmSampler(object):

    def __init__(self, output_folder, args_sampler,
                 observed_data, data_uncertainties,
                 observed_data_z_eval, sample_inds_1=None, sample_inds_2=None,
                 ignore_asymmetry=False, ignore_vz=False, **kwargs):

        # (self._tabpot, self._kde, self._phase_space_res) = args_sampler
        self._args_sampler = args_sampler

        self.output_folder = output_folder

        self._phase_space_res = args_sampler[-1]
        self._model_domain_1 = np.linspace(0, 2, self._phase_space_res)
        self._model_domain_2 = np.linspace(-2, 2, self._phase_space_res)
        model_domain = [self._model_domain_1, self._model_domain_2]

        self._prior_set = False
        self._data_domain = observed_data_z_eval
        self._observed_data = observed_data
        self._distance_calc = DistanceCalculator(model_domain, data_uncertainties, observed_data_z_eval,
                                                 sample_inds_1, sample_inds_2, ignore_asymmetry, ignore_vz)

    def set_prior(self, prior_class):

        self._prior_set = True
        self.priors_under_hood, self.priors_over_hood, self.to_sample_list = \
            prior_class.split_under_over_hood
        self.prior_class = prior_class
        self._dim = len(self.priors_over_hood)
        _ = self.bounds

    def run(self, n_particles=100, n_iterations=250, c1=0.5, c2=0.3, w=0.9, parallelize=False,
            n_proc=8):

        options = {'c1': c1, 'c2': c2, 'w': w}

        self._model_data = []
        # Call instance of PSO with bounds argument

        optimizer = pyswarms.single.GlobalBestPSO(n_particles=n_particles,
                                                  dimensions=self._dim,
                                                  options=options,
                                                  bounds=self.bounds)

        if parallelize:
            cost, pos = optimizer.optimize(self._minimize_func, iters=n_iterations,
                                           n_processes=n_proc)
        else:
            cost, pos = optimizer.optimize(self._minimize_func, iters=n_iterations)

        self._prior_set = False

        return cost, pos, optimizer.swarm, self._model_data

    def _minimize_func(self, parameters_sampled):

        logL = []
        for i in range(0, parameters_sampled.shape[0]):
            samples_prior_list = self._set_params(parameters_sampled[i,:])

            samples = {}
            for param_prior in samples_prior_list:
                param_name, value = self.prior_class.draw(param_prior)
                samples[param_name] = value

            asymmetry, mean_vz = single_iteration(samples, *self._args_sampler)
            model_data = [asymmetry, mean_vz]

            loglike = self._distance_calc.logLike(self._observed_data, model_data)
            logL.append(loglike)

            self._model_data.append((asymmetry, mean_vz, loglike))

        return np.absolute(logL)

    @property
    def bounds(self):

        if not hasattr(self, '_bounds'):
            _, priors_over_hood, _ = self.prior_class.split_under_over_hood
            bound_low, bound_high = [], []

            for param_prior in priors_over_hood:

                prior_type = param_prior[1]
                prior_args = param_prior[2]

                if prior_type != 'u':
                    raise Exception('only uniform priors allowed in PSO')

                bound_low.append(prior_args[0])
                bound_high.append(prior_args[1])

            bound_low, bound_high = np.array(bound_low), np.array(bound_high)
            self._bounds = (bound_low, bound_high)

        return self._bounds

    def _set_params(self, params_sampled):

        new_params = deepcopy(self.priors_under_hood)

        for j, param_name in enumerate(self.to_sample_list):

            new = [[param_name, 'f', params_sampled[j], False]]
            new_params += new
        return new_params
