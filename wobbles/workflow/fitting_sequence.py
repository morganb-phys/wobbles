from wobbles.Sampler.PMCABC import PMCABCSampler
from wobbles.Sampler.partcle_swarm import ParticleSwarmSampler
from wobbles.Sampler.mcmc import MCMCSampler
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
                output_list.append(['PSO', out, sampler])
                best_solution = np.array(out[1])

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

                out = sampler.run(**kwargs)

                output_list.append(['MCMC', out, sampler])
                best_index = np.argmax(out.log_prob)
                best_solution = out.coords[best_index]

            elif fit_type == 'PMCABC':

                sampler = PMCABCSampler(**self.kwargs_sampler)
                sampler.set_prior(prior)
                out = sampler.run(**kwargs)
                output_list.append(['PMCABC', out, sampler])
                distances = out.get_distances()
                parameters = np.squeeze(out.get_accepted_parameters())
                best_index = np.argmin(distances)
                best_solution = parameters[best_index]

            else:
                raise Exception('sampler type '+str(fit_type)+' not recognized')

        return output_list
