import emcee
import numpy as np
from wobbles.Sampler.data import DistanceCalculator
from copy import deepcopy
from wobbles.workflow.forward_model import single_iteration
import pickle
from wobbles.Sampler.base import MCMCBase
from multiprocessing import Pool

class MCMCSampler(MCMCBase):

    def __init__(self, output_folder, args_sampler,
                 observed_data, data_uncertainties,
                 observed_data_z_eval, sample_inds_1=None, sample_inds_2=None,
                 ignore_asymmetry=False, ignore_vz=False, **kwargs):

        # (self._tabpot, self._kde, self._phase_space_res) = args_sampler

        self.output_folder = output_folder

        self._phase_space_res = args_sampler[-1]
        self._model_domain_1 = np.linspace(0, 2, self._phase_space_res)
        self._model_domain_2 = np.linspace(-2, 2, self._phase_space_res)
        model_domain = [self._model_domain_1, self._model_domain_2]

        self._data_domain = observed_data_z_eval
        self._observed_data = observed_data
        self._distance_calc = DistanceCalculator(model_domain, data_uncertainties, observed_data_z_eval,
                                                 sample_inds_1, sample_inds_2, ignore_asymmetry, ignore_vz)

        super(MCMCSampler, self).__init__(output_folder, args_sampler,
                                          observed_data, observed_data_z_eval)


    def run(self, initial_pos, n_run, n_walkers_per_dim, save_output=True, progress=False,
            parallelize=False, n_proc=8):

        assert self._prior_set is True

        n_walkers = n_walkers_per_dim * self._dim

        if parallelize:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(n_walkers, self._dim, self.log_probability, pool=pool)
                state = sampler.run_mcmc(initial_pos, n_run, progress=progress);
            #pool.close()
        else:
            sampler = emcee.EnsembleSampler(n_walkers, self._dim, self.log_probability)
            state = sampler.run_mcmc(initial_pos, n_run, progress=progress);


        if save_output:
            save_name = self.output_folder + 'mcmc_chain'
            f = open(save_name, 'wb')
            pickle.dump(state, f)
            f.close()

        return state

    def log_probability(self, parameters_sampled):

        # check if sample should be rejected based on the prior
        log_prior_weight = self.log_prior(parameters_sampled)

        if not np.isfinite(log_prior_weight):
            return -np.inf

        asymmetry, mean_vz = self.model_data_from_params(parameters_sampled)

        model_data = [asymmetry, mean_vz]

        loglike = self._distance_calc.logLike(self._observed_data, model_data)

        return loglike
