from wobbles.workflow.forward_model import single_iteration
from wobbles.Sampler.data import *
from abcpy.continuousmodels import Uniform, Normal
from abcpy.probabilisticmodels import InputConnector, ProbabilisticModel, Continuous
import numpy as np
from copy import deepcopy
from abcpy.inferences import PMCABC
import os
import pickle
from wobbles.Sampler.base import Base
from abcpy.backends import BackendMPI, BackendDummy

class Simulator(ProbabilisticModel, Continuous):

    def __init__(self, mcmc_class, to_sample_list, priors_over_hood):

        self.to_sample_list = to_sample_list
        self.priors_over_hood = priors_over_hood
        self.mcmc_class = mcmc_class
        self._args_sampler = mcmc_class._args_sampler
        self._phase_space_res = mcmc_class._phase_space_res
        input_connector = InputConnector.from_list(self.formatted_prior)
        super().__init__(input_connector, 'DiskModel')

    def forward_simulate(self, params_sampled, k=1, rng=np.random.RandomState()):

        log_prior = self.mcmc_class.log_prior(params_sampled)
        phase_space_res = self.mcmc_class._phase_space_res

        if not np.isfinite(log_prior):
            asymmetry = 10000 * np.ones(phase_space_res)
            mean_vz = 10000 * np.ones_like(phase_space_res)

        else:
            samples_prior_list = self.mcmc_class._set_params(params_sampled)

            samples = {}
            for param_prior in samples_prior_list:
                param_name, value = self.mcmc_class.prior_class.draw(param_prior)
                samples[param_name] = value

            asymmetry, mean_vz = single_iteration(samples, *self._args_sampler)

            if asymmetry is None or mean_vz is None:
                # actually any array dim > 2 would work here
                asymmetry = 10000 * np.ones(self._phase_space_res)
                mean_vz = 10000 * np.ones_like(self._phase_space_res)

        return (asymmetry, mean_vz)

    @property
    def formatted_prior(self):

        param_prior_sampler = []

        for prior in self.priors_over_hood:

            prior_type = prior[1]
            param_name = prior[0]
            if prior_type == 'u':
                low, high = prior[2][0], prior[2][1]
                new_prior = Uniform([[low], [high]], name=param_name)
            elif prior_type == 'g':
                mean, sigma = prior[2][0], prior[2][1]
                new_prior = Normal([[mean], [sigma]], name=param_name)
            else:
                raise Exception('prior type ' + str(prior_type) + ' not valid')

            param_prior_sampler.append(new_prior)

        return param_prior_sampler

    def get_output_dimension(self):

        return len(self.to_sample_list)

    def _check_input(self, input_values):

        return True

    def _check_output(self, values):

        return True

class PMCABCSampler(Base):

    def __init__(self, output_folder, args_sampler,
                 observed_data, data_uncertainties,
                 observed_data_z_eval, sample_inds_1=None, sample_inds_2=None,
                 ignore_asymmetry=False, ignore_vz=False
                 ):

        self._obs = observed_data

        # (self._tabpot, self._kde, self._phase_space_res) = args_sampler
        self._args_sampler = args_sampler
        self._phase_space_res = args_sampler[-1]

        self.output_folder = output_folder

        model_domain_1 = np.linspace(0, 2, self._phase_space_res)
        model_domain_2 = np.linspace(-2, 2, self._phase_space_res)
        model_domain = [model_domain_1, model_domain_2]

        self._observed_data = observed_data
        self._distance_calc = DistanceCalculator(model_domain, data_uncertainties, observed_data_z_eval,
                 sample_inds_1, sample_inds_2, ignore_asymmetry, ignore_vz)

        super(PMCABCSampler, self).__init__(output_folder, args_sampler, observed_data,
                 observed_data_z_eval)

    def minimize_func(self, params):

        return self._distance_calc.distance(self._observed_data, params)

    def run(self, jobID, n_sample, steps, epsilon_init,
            epsilon_percentile, save_output=True, parallelize=False):

        assert self._prior_set is True

        if parallelize:
            backend = BackendMPI()
        else:
            backend = BackendDummy()

        steps_minus_1 = steps - 1
        epsilon_init = [epsilon_init] + [None] * steps_minus_1

        sim = Simulator(self, self.to_sample_list, self.priors_over_hood)
        sampler = PMCABC([sim], [self._distance_calc], backend, seed=1)

        journal_filename = self.output_folder + 'journal_' + jobID

        if os.path.exists(journal_filename):
            f = open(journal_filename, 'rb')
            journal_init = pickle.load(f)
            f.close()
            print('loading from journal file..')
            stat = journal_init.get_distances()
            print(str(epsilon_percentile) + 'th percentile of initial distances: ',
                  np.percentile(stat, epsilon_percentile))
        else:
            print('first_iteration...')
            journal_init = None

        journal = sampler.sample([self._obs], steps, epsilon_init, n_sample,
                                 1, epsilon_percentile, journal_class=journal_init)

        stat = journal.get_distances()
        print(str(epsilon_percentile) + 'th percentile of new distances: ', np.percentile(stat, epsilon_percentile))
        print('obtained ' + str(n_sample) + ' samples from ' + str(journal.number_of_simulations[0]) + ' realizations')

        if save_output:
            f = open(journal_filename, 'wb')
            pickle.dump(journal, f)
            f.close()

        self._prior_set = False

        return journal

