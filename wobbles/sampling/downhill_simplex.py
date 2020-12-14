import numpy as np
from copy import deepcopy
from wobbles.workflow.forward_model import single_iteration
from wobbles.sampling.data import DistanceCalculator
from scipy.optimize import minimize
from wobbles.sampling.base import Base

class DownhillSimplex(Base):

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

        super(DownhillSimplex, self).__init__(output_folder, args_sampler,
                                          observed_data, observed_data_z_eval)

    def minimize_func(self, params):

        log_prior_weight = self.prior_loglike(params)

        if not np.isfinite(log_prior_weight):
            return np.inf

        asymmetry, mean_vz, density = self.model_data_from_params(params)
        model_data = [asymmetry, mean_vz]
        stat = self._distance_calc.summary_statistic(self._observed_data, model_data)
        return stat

    def run(self, initial_pos, kwargs_optimizer={}, verbose=False):

        opt = minimize(self.minimize_func, x0=initial_pos, **kwargs_optimizer)
        if verbose:
            print(opt)

        return opt
