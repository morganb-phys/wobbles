from wobbles.Sampler.PMCABC import PMCABCSampler
from wobbles.Sampler.partcle_swarm import ParticleSwarmSampler
from wobbles.Sampler.mcmc import MCMCSampler
from wobbles.Sampler.downhill_simplex import DownhillSimplex
import numpy as np

class OutputContainer(object):

    def __init__(self, output):

        self.output = output

class FittingSequence(object):

    def __init__(self, kwargs_sampler):

        self.kwargs_sampler = kwargs_sampler

    def fitting_sequence(self, sequence_list):

        output_list = []
        best_solution = None

        for iteration in sequence_list:

            fit_type = iteration[0]
            kwargs = iteration[1]
            prior = iteration[2]

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
                    assert 'initial_pos' in kwargs.keys()
                else:
                    shape = (len(best_solution) * kwargs['n_walkers_per_dim'], len(best_solution))
                    sigmas = np.absolute(kwargs['init_scale'] * best_solution)
                    initial_pos = np.empty(shape)

                    for i in range(0, initial_pos.shape[0]):
                        initial_pos[i,:] = np.random.normal(best_solution, sigmas)
                    kwargs['initial_pos'] = initial_pos
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

        return output_list
