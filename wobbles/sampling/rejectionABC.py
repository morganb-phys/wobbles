from wobbles.workflow.forward_model import single_iteration
from wobbles.sampling.data import Data
import numpy as np
from time import time

class SimulationContainer(object):

    def __init__(self, z_domain, simulated_asymmetry, simulated_mean_vz, simulated_params):

        self.simulated_asymmetry = simulated_asymmetry
        self.simulated_mean_vz = simulated_mean_vz
        self.simulated_params = simulated_params
        self.z_domain = z_domain

    def _likelihood(self, z_observed, observed, simulated, uncertainties, model_percent_uncertainty):

        data = Data(z_observed, observed, uncertainties)

        N = simulated.shape[0]
        n_points = simulated.shape[1]
        chi_square = np.empty(N)

        for i in range(0, N):

            if model_percent_uncertainty is None:
                model_uncertainties = np.zeros(n_points)
            else:
                model_uncertainties = model_percent_uncertainty * simulated[i, :]

            x = simulated[i, :] * (1 + model_uncertainties)
            chi_square[i] = data.chi_square(self.z_domain, x)

        return chi_square

    def likelihood_asymmetry(self, z_observed, observed_asymmetry, uncertainties, model_percent_uncertainty=None):

        chi_square = self._likelihood(z_observed, observed_asymmetry, self.simulated_asymmetry, uncertainties,
                                      model_percent_uncertainty)

        self.chi_square_asymmetry = chi_square

        return chi_square

    def likelihood_meanvz(self, z_observed, observed_meanvz, uncertainties, model_percent_uncertainty=None):

        chi_square = self._likelihood(z_observed, observed_meanvz, self.simulated_mean_vz, uncertainties,
                                      model_percent_uncertainty)

        self.chi_square_meanvz = chi_square

        return chi_square


class RejectionABCSampler(object):

    def __init__(self, prior_class, output_folder, args_sampler, run_index, Nrealizations,
                 readout_steps=25, parallel=False, n_proc=None, kwargs_sampler={}):

        # (self._tabpot, self._kde, self._phase_space_res) = args_sampler
        self._args_sampler = args_sampler
        self._phase_space_dim = args_sampler[-1]
        self.run_index = run_index
        self.output_folder = output_folder
        self.Nrealizations = Nrealizations
        self.prior_class = prior_class
        self.readout_steps = readout_steps
        self.parallel = parallel
        self.n_proc = n_proc
        self._kwargs_sampler = kwargs_sampler

    def run(self, save_output=True, pool=None, verbose=False):

        init_arrays = True
        count = 0
        if self.parallel:
            assert self.n_proc is not None, 'If running with multiprocessing must specify number ' \
                                            'of parallel processes'
            n_run = int(self.Nrealizations / self.n_proc)
            readout_steps = int(self.readout_steps / self.n_proc)
            if verbose:
                print('running with multiproccessing... ')
                print(str(n_run) + ' iterations total and ' + str(self.n_proc) + ' jobs per iteration')

        else:

            n_run = self.Nrealizations
            readout_steps = self.readout_steps
            if verbose:
                print('running without multiproccessing... ')
                print(str(n_run) + ' iterations total')

        for j in range(0, n_run):

            parameter_priors = self.prior_class.param_prior
            _, _, save_params_list = self.prior_class.split_under_over_hood

            if self.parallel:

                t0 = time()
                A, vz, density, new_params_sampled = self._run_pool(parameter_priors, save_params_list, pool)
                dt = np.round((time() - t0) / self.n_proc, 2)
                if verbose:
                    print('sampling rate: ' + str(dt) + ' seconds per iteration')

            else:
                t0 = time()
                A, vz, density, new_params_sampled = self._run(parameter_priors, save_params_list)
                dt = np.round(time() - t0, 2)
                if verbose:
                    print('sampling rate: ' + str(dt) + ' seconds per iteration')

            if init_arrays:
                init_arrays = False
                params_sampled = new_params_sampled
                asymmetry = A
                mean_vz = vz
                rho = density
            else:
                params_sampled = np.vstack((params_sampled, new_params_sampled))
                asymmetry = np.vstack((asymmetry, A))
                mean_vz = np.vstack((mean_vz, vz))
                rho = np.vstack((rho, density))

            count += 1
            if save_output and count < readout_steps:
                readout = False
            else:
                readout = True

            if readout and save_output:

                info = {}
                for param in parameter_priors:
                    name = param[0]
                    prior_type = param[1]

                    if prior_type == 'u':
                        ran = [param[2][0], param[2][1]]
                    elif prior_type == 'g':
                        ran = [param[2][0] - 3 * param[2][1], param[2][0] + 3 * param[2][1]]
                    else:
                        continue

                    info[name] = ran
                with open(self.output_folder + 'param_names_ranges.txt', 'w') as f:
                    f.write(str(info))

                init_arrays = True
                count = 0
                with open(self.output_folder + 'asymmetry_' + str(self.run_index) + '.txt', 'a') as f:
                    string_to_write = ''
                    for row in range(0, asymmetry.shape[0]):
                        for ai in asymmetry[row, :]:
                            string_to_write += str(np.round(ai, 5)) + ' '
                        string_to_write += '\n'
                    f.write(string_to_write)

                with open(self.output_folder + 'meanvz_' + str(self.run_index) + '.txt', 'a') as f:
                    string_to_write = ''
                    for row in range(0, mean_vz.shape[0]):
                        for vzi in mean_vz[row, :]:
                            string_to_write += str(np.round(vzi, 5)) + ' '
                        string_to_write += '\n'
                    f.write(string_to_write)

                with open(self.output_folder + 'density_' + str(self.run_index) + '.txt', 'a') as f:
                    string_to_write = ''
                    for row in range(0, rho.shape[0]):
                        for rhoi in rho[row, :]:
                            string_to_write += str(np.round(rhoi, 5)) + ' '
                        string_to_write += '\n'
                    f.write(string_to_write)

                with open(self.output_folder + 'params_' + str(self.run_index) + '.txt', 'a') as f:
                    string_to_write = ''
                    for row in range(0, params_sampled.shape[0]):
                        for param_val in params_sampled[row, :]:
                            string_to_write += str(np.round(param_val, 5)) + ' '
                        string_to_write += '\n'
                    f.write(string_to_write)

        if save_output is False:
            return asymmetry, mean_vz, params_sampled

    def _run_pool(self, parameter_priors, save_params_list, pool):

        samples_list = []

        new_params_sampled = np.empty((self.n_proc, len(save_params_list)))

        for i in range(0, self.n_proc):

            samples = self.prior_class.draw(parameter_priors)
            samples_list.append(samples)

            new_params = [samples[param] for param in save_params_list]
            new_params_sampled[i, :] = np.array(new_params)


        model_data = list(pool.map(self._func, samples_list))

        A_len = len(model_data[0][0])
        vz_len = len(model_data[0][1])
        density_len = len(model_data[0][2])
        A, vz = np.empty((self.n_proc, A_len)), np.empty((self.n_proc, vz_len))
        rho = np.empty((self.n_proc, density_len))

        for i, di in enumerate(model_data):

            if model_data[i][0] is None or model_data[i][1] is None:
                model_data[i][0] = np.ones(len(self._phase_space_dim)) * 1000
                model_data[i][1] = np.ones(len(self._phase_space_dim)) * 1000
                model_data[i][2] = np.ones(len(self._phase_space_dim)) * 1000

            A[i, :] = model_data[i][0]
            vz[i,:] = model_data[i][1]
            rho[i,:] = model_data[i][2]

        return A, vz, rho, new_params_sampled

    def _func(self, x):

        A, vz, rho = single_iteration(x, *self._args_sampler, **self._kwargs_sampler)

        return (A, vz, rho)

    def _run(self, parameter_priors, save_params_list):

        samples = self.prior_class.draw(parameter_priors)

        for param in save_params_list:
            assert param in samples.keys()

        A, vz, rho = single_iteration(samples, *self._args_sampler, **self._kwargs_sampler)

        if A is None or vz is None:
            A = np.ones(self._phase_space_dim) * 1000
            vz = np.ones(self._phase_space_dim) * 1000
            rho = np.ones(self._phase_space_dim) * 1000

        new_params_sampled = [samples[param] for param in save_params_list]
        new_params_sampled = np.array(new_params_sampled)

        return A, vz, rho, new_params_sampled



