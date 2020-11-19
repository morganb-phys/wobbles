from wobbles.sampling.samplers.pso_sampler import ParticleSwarmOptimizer
import numpy as np
from copy import deepcopy
from wobbles.workflow.forward_model import single_iteration
from wobbles.sampling.data import DistanceCalculator
from wobbles.sampling.base import Base

class ParticleSwarmSampler(Base):

    def __init__(self, output_folder, args_sampler,
                 observed_data, data_uncertainties,
                 observed_data_z_eval, sample_inds_1=None, sample_inds_2=None,
                 ignore_asymmetry=False, ignore_vz=False, **kwargs):

        self._phase_space_res = args_sampler[-1]
        self._model_domain_1 = np.linspace(0, 2, self._phase_space_res)
        self._model_domain_2 = np.linspace(-2, 2, self._phase_space_res)
        model_domain = [self._model_domain_1, self._model_domain_2]

        self._data_domain = observed_data_z_eval
        self._observed_data = observed_data
        self._distance_calc = DistanceCalculator(model_domain, data_uncertainties, observed_data_z_eval,
                                                 sample_inds_1, sample_inds_2, ignore_asymmetry, ignore_vz)

        super(ParticleSwarmSampler, self).__init__(output_folder, args_sampler,
                                          observed_data, observed_data_z_eval)

    def run(self, n_particles=100, n_iterations=250, c1=1.193, c2=1.193,
            parallelize=False, pool=None, verbose=False):

        bounds = self.bounds

        if parallelize:

            assert pool is not None

            pso = ParticleSwarmOptimizer(self.minimize_func, bounds[0], bounds[1],
                                            n_particles, pool=pool)

        else:
            pso = ParticleSwarmOptimizer(self.minimize_func, bounds[0], bounds[1],
                      n_particles)

        gbest, [chi2_list, pos_list, vel_list] = pso.optimize(max_iter=n_iterations, verbose=verbose, c1=c1, c2=c2)

        self._prior_set = False

        return (gbest, np.min(chi2_list), pso.swarm)

    @property
    def bounds(self):

        if not hasattr(self, '_pso_bounds'):

            _, priors_over_hood, _ = self.prior_class.split_under_over_hood
            bound_low, bound_high = [], []

            for param_prior in priors_over_hood:

                prior_type = param_prior[1]
                prior_args = param_prior[2]

                if prior_type == 'u':

                    bound_low.append(prior_args[0])
                    bound_high.append(prior_args[1])

                elif prior_type == 'g':

                    mu, sig = prior_args[0], prior_args[1]
                    bound_low.append(mu - sig)
                    bound_high.append(mu + sig)

                else:
                    raise Exception('prior type ' + str(prior_type) + ' not recognized')

            self._pso_bounds = [bound_low, bound_high]

        return self._pso_bounds

    def minimize_func(self, parameters_sampled):

        # check if sample should be rejected based on the prior
        chi_square_prior = -2 * self.prior_loglike(parameters_sampled)

        if not np.isfinite(chi_square_prior):
            return 1e+9

        asymmetry, mean_vz = self.model_data_from_params(parameters_sampled)

        model_data = [asymmetry, mean_vz]

        chi_square_model = self._distance_calc.chi_square(self._observed_data, model_data)

        return chi_square_model + chi_square_prior
