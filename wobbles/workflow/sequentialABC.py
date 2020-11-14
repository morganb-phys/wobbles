from wobbles.workflow.forward_model_subs import single_iteration
import os
import sys
import pickle
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.continuousmodels import Uniform

from abcpy.backends import BackendDummy as Backend
from abcpy.inferences import RejectionABC, PMCABC

class Data(object):

    def __init__(self, z_observed, data, errors, sample_inds=None):

        assert len(z_observed) == len(data)
        assert len(errors) == len(data)

        if sample_inds is not None:
            z_observed = np.array(z_observed)[sample_inds]
            data = np.array(data)[sample_inds]
            errors = np.array(errors)[sample_inds]

        self.zobs = z_observed
        self.data = data
        self.errors = errors

    def chi_square(self, z_model, model, error_inflate=1.):

        interp_model = interp1d(z_model, model)

        exponent = 0
        for (z_i, data_i, error_i) in zip(self.zobs, self.data, self.errors):
            delta = interp_model(z_i) - data_i
            dx = delta / (error_i * error_inflate)
            exponent += dx ** 2

        ndof = len(self.data) - 1
        return exponent / ndof


class JointData(object):

    def __init__(self, data_1, data_2, ignore_1=False, ignore_2=False):

        self.data_1, self.data_2 = data_1, data_2

        self.ignore_1 = ignore_1
        self.ignore_2 = ignore_2

    def penalty(self, z_model_1, z_model_2, model_1, model_2):

        n = 0
        if self.ignore_1:
            pen1 = 0
        else:
            n += 1
            pen1 = self.data_1.chi_square(z_model_1, model_1)
        if self.ignore_2:
            pen2 = 0
        else:
            n += 1
            pen2 = self.data_2.chi_square(z_model_2, model_2)
        assert n > 0
        return pen1 + pen2

class DiskModel(ProbabilisticModel, Continuous):

    def __init__(self, parameter_priors, to_sample_list, param_priors, args_sampler,
                 name='DiskModel', phase_space_res=60):

        self.to_sample_list = to_sample_list
        self.param_prior = param_priors
        self.args_sampler = args_sampler

        self._domain_1_model = np.linspace(0, 2, phase_space_res)
        self._domain_2_model = np.linspace(-2, 2, phase_space_res)
        self.phase_space_res = phase_space_res

        args = [arg for arg in args_sampler] + [phase_space_res]
        self._args_sampler = tuple(args)

        if not isinstance(parameter_priors, list):
            raise TypeError('Input of Normal model is of type list')

        input_connector = InputConnector.from_list(parameter_priors)
        super().__init__(input_connector, name)

    def forward_simulate(self, params_sampled, k=1, rng=np.random.RandomState()):

        samples_prior_list = self._set_params(params_sampled)

        samples = self._array_to_dictionary(samples_prior_list)

        asymmetry, mean_vz = single_iteration(samples, *self._args_sampler)

        if asymmetry is None or mean_vz is None:
            asymmetry = 10000 * np.ones_like(self._domain_1_model)
            mean_vz = 10000 * np.ones_like(self._domain_2_model)

        return (asymmetry, mean_vz)

    def _check_input(self, input_values):

        return True

    def _check_output(self, values):

        return True

    def _set_params(self, params_sampled):

        new_params = deepcopy(self.param_prior)
        for j, param_name in enumerate(self.to_sample_list):
            new = [[param_name, 'f', params_sampled[j], False]]
            new_params += new
        return new_params

    def get_output_dimension(self):
        return len(self.to_sample_list)

    @staticmethod
    def _array_to_dictionary(p):

        samples = {}
        for prior in p:

            param_name = prior[0]
            prior_type = prior[1]
            prior_args = prior[2]
            positive_definite = prior[3]
            if prior_type == 'g':
                value = np.random.normal(*prior_args)
            elif prior_type == 'u':
                value = np.random.uniform(*prior_args)
            elif prior_type == 'f':
                value = prior_args
            else:
                raise Exception('param prior ' + str(prior[0]) + ' not valid.')
            if positive_definite:
                value = abs(value)
            samples[param_name] = value

        return samples


class DistanceCalculator(object):

    def __init__(self, simulator, data_uncertainties, data_domain,
                 sample_inds_1=None, sample_inds_2=None, ignore_1=False, ignore_2=False):
        self.simulator = simulator
        self.data_uncertainties = data_uncertainties
        self.data_domain = data_domain
        self.sample_inds_1 = sample_inds_1
        self.sample_inds_2 = sample_inds_2
        self.ignore1, self.ignore2 = ignore_1, ignore_2

    def distance(self, observed_data, model_data):
        data_1 = Data(self.data_domain[0], observed_data[0], self.data_uncertainties[0], self.sample_inds_1)
        data_2 = Data(self.data_domain[1], observed_data[1], self.data_uncertainties[1], self.sample_inds_2)
        joint_data = JointData(data_1, data_2, self.ignore1, self.ignore2)

        (asymmetry, mean_vz) = model_data
        rho = joint_data.penalty(self.simulator._domain_1_model, self.simulator._domain_2_model, asymmetry, mean_vz)

        return rho

    def dist_max(self):
        return 1e+3

# import numpy as np
# import pickle
# from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
# from abcpy.continuousmodels import Uniform
# from abcpy.inferences import PMCABC
#
# data_path = os.getenv('HOME') + '/Code/jupyter_notebooks/disk_dynamics/'
# A = np.loadtxt(data_path+'Ap_fake_data_0.txt')
# z_data_a, data_asymmetry = A[2,:], A[3,:]
# delta_a = [0.1] * len(z_data_a)
# VZ = np.loadtxt(data_path+'vz_fake_data_0.txt')
# z_data_vz, data_meanvz = VZ[0,:], VZ[1,:]
# delta_meanvz = [0.25] * len(z_data_vz)
#
# # data_path = os.getenv('HOME') + '/data/'
# # A = np.loadtxt(data_path+'Ap_fake_data_0.txt')
# # z_data_a, data_asymmetry = A[2,:], A[3,:]
# # delta_a = [max(0.025, 0.05 * da) for da in data_asymmetry]
# # VZ = np.loadtxt(data_path+'vz_fake_data_0.txt')
# # z_data_vz, data_meanvz = VZ[0,:], VZ[1,:]
# # delta_meanvz = [max(0.02, 0.05 * v) for v in data_meanvz]
#
# output_folder = './'
# VLA_data_path = os.getenv('HOME') + '/Code/external/wobbles/wobbles/workflow/'
#
# f = open(VLA_data_path+'saved_potentials/tabulated_MWpot', "rb")
# tabulated_potential = pickle.load(f)
# f.close()
# vla_subhalo_phase_space = np.loadtxt(VLA_data_path + 'vl2_halos_scaled.dat')
# # kde_instance = gaussian_kde(vla_subhalo_phase_space, bw_method=0.1)
# kde_instance = None
# args_sampler = (tabulated_potential, kde_instance)
#
# param_prior = []
# # param_prior += [['nfw_norm', 'u', [0.15, 0.45], False]]
# # param_prior += [['disk_norm', 'u', [0.5, 0.7], False]]
# # param_prior += [['log_sag_mass_DM', 'u', [8.7, 11], False]]
# param_prior += [['sag_mass2light', 'g', [50, 5], True]]
# param_prior += [['f_sub', 'f', 0., False]]
# param_prior += [['log_slope', 'f', -1.9, False]]
# param_prior += [['m_host', 'f', 1.2e+12, False]]
# # param_prior += [['velocity_dispersion_1', 'u', [15, 25], False]]
# param_prior += [['component_amplitude_1', 'f', 0.5, False]]
# #param_prior += [['velocity_dispersion_2', 'f', None, False]]
# param_prior += [['component_amplitude_2', 'f', 0.5, False]]
# param_prior += [['velocity_dispersion_3', 'f', None, False]]
# param_prior += [['component_amplitude_3', 'f', None, False]]
# #param_prior += [['orbit_ra', 'u', [283-10, 283+10], False]]
# #param_prior += [['orbit_dec', 'u', [-30-5, -30+5], False]]
# #param_prior += [['orbit_z', 'u', [21, 23], False]]
# param_prior += [['orbit_pm_ra', 'f', -2.6, False]]
# param_prior += [['orbit_pm_dec', 'f', -1.3, False]]
# #param_prior += [['orbit_vlos', 'u', [140-5, 140+5], False]]
# #param_prior += [['orbit_ra', 'f', 220, False]]
# #param_prior += [['orbit_dec', 'f', -50, False]]
# #param_prior += [['orbit_z', 'f', 22, False]]
# #param_prior += [['orbit_pm_ra', 'f', -3.1, False]]
# #param_prior += [['orbit_pm_dec', 'f', -2.8, False]]
# #param_prior += [['orbit_vlos', 'f', 140, False]]
# #param_prior += [['gal_norm', 'u', [0.7, 1.3], False]]
#
#
# to_sample_list = ['nfw_norm', 'disk_norm', 'log_sag_mass_DM',
#                   'velocity_dispersion_1', 'velocity_dispersion_2', 'gal_norm',
#                   'orbit_ra', 'orbit_dec', 'orbit_z', 'orbit_vlos']
#
# p1 = Uniform([[0.15], [0.45]], name="nfw_norm")
# p2 = Uniform([[0.5], [0.7]], name="disk_norm")
# p3 = Uniform([[8.5], [11]], name="log_Msat")
# p4 = Uniform([[15], [25]], name="vdis_1")
# p5 = Uniform([[5], [10]], name="vdis_2")
# p6 = Uniform([[0.7], [1.3]], name="gal_norm")
# p7 = Uniform([[273], [293]], name="orbit_ra")
# p8 = Uniform([[-40], [-20]], name="orbit_dec")
# p9 = Uniform([[25], [27]], name="orbit_z")
# p10 = Uniform([[135], [145]], name="orbit_vlos")
# param_prior_sampler = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
# params_sample_internal = [p[0] for p in param_prior]
# assert len(param_prior_sampler) == len(to_sample_list)
# for pname in to_sample_list:
#     assert pname not in params_sample_internal
#
# sim = DiskModel(param_prior_sampler, to_sample_list, param_prior, args_sampler)
#
# from abcpy.backends import BackendDummy as Backend
# backend = Backend()
# sample_inds_1 = None
# sample_inds_2 = None
# ignore_asymmetry = False
# ignore_vz = True
# distance_calculator = DistanceCalculator(sim, [np.array(delta_a), np.array(delta_meanvz)],
#                                          [z_data_a, z_data_vz], sample_inds_1, sample_inds_2,
#                                          ignore_asymmetry, ignore_vz)
# obs = [data_asymmetry, data_meanvz]
# sampler = PMCABC([sim], [distance_calculator], backend, seed=1)
#
# n_sample, n_samples_per_param = 2, 1
# steps = 3
# epsilon_init = np.linspace(50, 100, steps)[::-1]
# print(epsilon_init)
#journal = sampler.sample([obs], steps, epsilon_init, n_sample, n_samples_per_param)
# print(journal.number_of_simulations)
# stats = journal.get_distances()
# posterior_samples = np.array(journal.get_accepted_parameters()).squeeze()
# print(stats)
# print(posterior_samples)
# f = open(output_folder + 'testjournal', 'wb')
# pickle.dump(journal, f)
# f.close()



# from wobbles.workflow.forward_model_subs import single_iteration
# import numpy as np
# from copy import deepcopy
# from scipy.interpolate import interp1d
# import astroabc

# class Data(object):
#
#     def __init__(self, z_observed, data, errors):
#         assert len(z_observed) == len(data)
#         assert len(errors) == len(data)
#         self.zobs = z_observed
#         self.data = data
#         self.errors = errors
#
#     def penalty(self, z_model, model, error_inflate=1.):
#         return np.sqrt(self.chi_square(z_model, model, error_inflate))
#
#     def chi_square(self, z_model, model, error_inflate=1.):
#         interp_model = interp1d(z_model, model)
#
#         exponent = 0
#         for (z_i, data_i, error_i) in zip(self.zobs, self.data, self.errors):
#             delta = interp_model(z_i) - data_i
#             dx = delta / (error_i * error_inflate)
#             exponent += dx ** 2
#
#         return exponent / len(z_model)
#
#
# class JointData(object):
#
#     def __init__(self, data_1, data_2, sigma_1=1., sigma_2=1.):
#
#         self.data_1, self.data_2 = data_1, data_2
#
#         self.sigma_1 = sigma_1
#         self.sigma_2 = sigma_2
#
#     def penalty(self, z_model_1, z_model_2, model_1, model_2):
#
#         if self.sigma_1 is None:
#             pen1 = 0
#         else:
#             pen1 = self.data_1.penalty(z_model_1, model_1, self.sigma_1)
#         if self.sigma_2 is None:
#             pen2 = 0
#         else:
#             pen2 = self.data_2.penalty(z_model_2, model_2, self.sigma_2)
#         return pen1 + pen2
#
#
# class Simulation(object):
#
#     def __init__(self, to_sample_list, param_priors, args_sampler):
#
#         self.to_sample_list = to_sample_list
#         self.param_prior = param_priors
#         self.args_sampler = args_sampler
#
#         self._z1 = np.linspace(0, 2, 100)
#         self._z2 = np.linspace(-2, 2, 100)
#
#     def _set_params(self, params_sampled):
#
#         new_params = deepcopy(self.param_prior)
#         for j, param_name in enumerate(self.to_sample_list):
#             new_prior = [[param_name, 'f', params_sampled[j], False]]
#             new_params += new_prior
#         return new_params
#
#     @staticmethod
#     def _array_to_dictionary(parameter_priors):
#
#         samples = {}
#         for param_prior in parameter_priors:
#
#             param_name = param_prior[0]
#             prior_type = param_prior[1]
#             prior_args = param_prior[2]
#             positive_definite = param_prior[3]
#             if prior_type == 'g':
#                 value = np.random.normal(*prior_args)
#             elif prior_type == 'u':
#                 value = np.random.uniform(*prior_args)
#             elif prior_type == 'f':
#                 value = prior_args
#             else:
#                 raise Exception('param prior ' + str(param_prior[0]) + ' not valid.')
#             if positive_definite:
#                 value = abs(value)
#             samples[param_name] = value
#
#         return samples
#
#     def simulate(self, params_sampled):
#
#         samples_prior_list = self._set_params(params_sampled)
#
#         samples = self._array_to_dictionary(samples_prior_list)
#
#         asymmetry, mean_vz = single_iteration(samples, *self.args_sampler)
#
#         if asymmetry is None or mean_vz is None:
#             asymmetry = 10000 * np.ones_like(self._z1)
#             mean_vz = 10000 * np.ones_like(self._z2)
#
#         return (asymmetry, mean_vz)
#
#     def dfunc(self, joint_data_class, model):
#
#         (asymmetry, mean_vz) = model
#         rho = joint_data_class.penalty(self._z1, self._z2, asymmetry, mean_vz)
#         return rho