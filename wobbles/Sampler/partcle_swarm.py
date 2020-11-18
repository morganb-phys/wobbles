from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from lenstronomy.Sampling.Pool.pool import choose_pool
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

    def run(self, n_particles=100, n_iterations=250, c1=1.193, c2=1.193,
            parallelize=False, pool=None, verbose=False):

        if parallelize:

            assert pool is not None

            pso = ParticleSwarmOptimizer(self.minimize_func, self.bounds[0], self.bounds[1],
                                            n_particles, pool=pool)

        else:
            pso = ParticleSwarmOptimizer(self.minimize_func, self.bounds[0], self.bounds[1],
                      n_particles)

        swarm, gbest = pso.optimize(max_iter=n_iterations, verbose=verbose, c1=c1, c2=c2)

        self._prior_set = False

        return swarm, gbest

    def model_data_from_params(self, parameters_sampled):

        samples_prior_list = self._set_params(parameters_sampled)

        samples = {}
        for param_prior in samples_prior_list:
            param_name, value = self.prior_class.draw(param_prior)
            samples[param_name] = value

        asymmetry, mean_vz = single_iteration(samples, *self._args_sampler)

        return asymmetry, mean_vz

    def minimize_func(self, parameters_sampled):

        asymmetry, mean_vz = self.model_data_from_params(parameters_sampled)

        if asymmetry is None or mean_vz is None:
            # actually any array dim > 2 would work here

            loglike = -np.inf

        else:
            model_data = [asymmetry, mean_vz]

            loglike = self._distance_calc.logLike(self._observed_data, model_data)

        return loglike

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
