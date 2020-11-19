from wobbles.sampling.PMCABC import PMCABCSampler
from wobbles.sampling.partcle_swarm import ParticleSwarmSampler
from wobbles.sampling.mcmc import MCMCSampler
from wobbles.sampling.downhill_simplex import DownhillSimplex
import numpy as np

class OutputContainer(object):

    def __init__(self, output, kwargs_sampler):

        self.output = output
        self.kwargs_sampler = kwargs_sampler

class FittingSequence(object):

    def __init__(self, kwargs_sampler):

        self.kwargs_sampler = kwargs_sampler

    @classmethod
    def from_saved_output(cls, output_container):

        """
        Restart a new fitting sequence from output stored in Output Container
        :param output_container: instance of ^
        :return:
        """

        kwargs_sampler = output_container.kwargs_sampler
        return FittingSequence(kwargs_sampler)

    @staticmethod
    def mcmc_initial_pos_from_best(params_best, kwargs):

        init_scale = kwargs['init_scale']
        shape = (len(params_best) * kwargs['n_walkers_per_dim'], len(params_best))
        sigmas = np.absolute(init_scale * params_best)
        initial_pos = np.empty(shape)
        for i in range(0, initial_pos.shape[0]):
            initial_pos[i, :] = np.random.normal(params_best, sigmas)
        return initial_pos

    def fitting_sequence(self, sequence_list, verbose=False):

        output_list = []
        best_solution = None

        for iteration in sequence_list:

            fit_type = iteration[0]
            kwargs = iteration[1]
            prior = iteration[2]

            if verbose:
                print('running '+fit_type+'... ')

            if fit_type == 'PSO':

                sampler = ParticleSwarmSampler(**self.kwargs_sampler)
                sampler.set_prior(prior)
                out = sampler.run(**kwargs)
                best_solution = np.array(out[0])
                best_chi2 = out[1]
                swarm = out[2]
                output_list.append(['PSO', (best_solution, best_chi2), swarm, sampler])

            elif fit_type == 'MCMC':

                sampler = MCMCSampler(**self.kwargs_sampler)
                sampler.set_prior(prior)
                if best_solution is None:
                    assert 'initial_position' in kwargs.keys()
                    kwargs['initial_pos'] = self.mcmc_initial_pos_from_best(kwargs['initial_position'],
                                                                            kwargs)
                    del kwargs['init_scale']
                    del kwargs['initial_position']

                else:
                    kwargs['initial_pos'] = self.mcmc_initial_pos_from_best(best_solution, kwargs)
                    del kwargs['init_scale']

                chain = sampler.run(**kwargs)
                best_index = np.argmax(chain.log_prob)
                best_solution = chain.coords[best_index]
                best_chi2 = -2 * sampler.log_probability(best_solution)
                output_list.append(['MCMC', (best_solution, best_chi2), chain, sampler])

            elif fit_type == 'PMCABC':

                sampler = PMCABCSampler(**self.kwargs_sampler)
                sampler.set_prior(prior)
                journal = sampler.run(**kwargs)
                distances = out.get_distances()
                parameters = np.squeeze(out.get_accepted_parameters())
                best_index = np.argmin(distances)
                best_solution = parameters[best_index]
                best_chi2 = sampler.minimize_func(best_solution)
                output_list.append(['PMCABC', (best_solution, best_chi2), journal, sampler])

            elif fit_type == 'AMOEBA':

                sampler = DownhillSimplex(**self.kwargs_sampler)
                sampler.set_prior(prior)
                out = sampler.run(**kwargs)
                best_solution = out['x']
                best_stat = sampler.minimize_func(best_solution)
                output_list.append(['AMOEBA', (best_solution, best_stat), out, sampler])

            else:
                raise Exception('sampler type '+str(fit_type)+' not recognized')

            if verbose:
                print('found best result: ', (best_solution, output_list[-1][1][1]))
                print('\n')

        return output_list
