from wobbles.workflow.forward_model import single_iteration
import numpy as np


class RejectionABCSampler(object):

    def __init__(self, prior_class, output_folder, args_sampler, run_index, Nrealizations,
                 readout_steps=25, pool=None, n_proc=None):

        # (self._tabpot, self._kde, self._phase_space_res) = args_sampler
        self._args_sampler = args_sampler

        self.run_index = run_index
        self.output_folder = output_folder
        self.Nrealizations = Nrealizations
        self.prior_class = prior_class
        self.readout_steps = readout_steps
        self.pool = pool
        self.n_proc = n_proc

    def run(self, save_output=True):

        init_arrays = True
        count = 0
        if self.pool is None:
            n_run = self.Nrealizations
        else:
            assert self.n_proc is not None, 'If running with multiprocessing must specify number ' \
                                            'of parallel processes'
            n_run = int(self.Nrealizations/self.n_proc)

        for j in range(0, n_run):

            parameter_priors = self.prior_class.param_prior
            _, _, save_params_list = self.prior_class.split_under_over_hood

            if self.pool is None:
                A, vz, new_params_sampled = self._run(parameter_priors, save_params_list)
            else:
                A, vz, new_params_sampled = self._run_pool(parameter_priors, save_params_list)

            if init_arrays:
                init_arrays = False
                params_sampled = new_params_sampled
                asymmetry = A
                mean_vz = vz
            else:
                params_sampled = np.vstack((params_sampled, new_params_sampled))
                asymmetry = np.vstack((asymmetry, A))
                mean_vz = np.vstack((mean_vz, vz))

            count += 1
            if save_output and count < self.readout_steps:
                readout = False
            else:
                readout = True

            if save_output is False:
                return asymmetry, mean_vz, params_sampled

            if readout and save_output:

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

                with open(self.output_folder + 'params_' + str(self.run_index) + '.txt', 'a') as f:
                    string_to_write = ''
                    for row in range(0, params_sampled.shape[0]):
                        for param_val in params_sampled[row, :]:
                            string_to_write += str(np.round(param_val, 5)) + ' '
                        string_to_write += '\n'
                    f.write(string_to_write)

    def _run_pool(self, parameter_priors, save_params_list):

        samples_list = []
        new_params_sampled = np.empty(self.n_proc, len(save_params_list))

        for i in range(0, self.n_proc):
            samples = {}
            for param_prior in parameter_priors:
                param_name, value = self.prior_class.draw(param_prior)
                samples[param_name] = value
            samples_list.append(samples)

            new_params = [samples[param] for param in save_params_list]
            new_params_sampled[i, :] = np.array(new_params)

        model_data = list(self.pool.map(self._func, samples_list))

        A_shape = model_data[0][0].shape
        vz_shape = model_data[0][1].shape
        A, vz = np.empty(A_shape), np.empty(vz_shape)

        for i, di in enumerate(model_data):
            A[i, :] = model_data[i][0]
            vz[i,:] = model_data[i][1]

        return A, vz, new_params_sampled

    def _func(self, x):

        A, vz = single_iteration(x, *self._args_sampler)
        return (A, vz)

    def _run(self, parameter_priors, save_params_list):

        samples = {}
        for param_prior in parameter_priors:

            param_name, value = self.prior_class.draw(param_prior)
            samples[param_name] = value

        for param in save_params_list:
            assert param in samples.keys()

        A, vz = single_iteration(samples, *self._args_sampler)

        new_params_sampled = [samples[param] for param in save_params_list]
        new_params_sampled = np.array(new_params_sampled)

        return A, vz, new_params_sampled



