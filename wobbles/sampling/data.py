import numpy as np
from scipy.interpolate import interp1d

class DataContainer(object):

    def __init__(self, z_a, a, delta_a, z_vz, v_z, delta_v_z, true_params=None):
        self.data_asymmetry = [z_a, a, delta_a]
        self.data_vz = [z_vz, v_z, delta_v_z]

        self.true_params = true_params

        self.observed_data = [a, v_z]
        self.data_uncertainties = [delta_a, delta_v_z]
        self.observed_data_z_eval = [z_a, z_vz]

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

    def summary_statistic(self, z_model, model):

        interp_model = interp1d(z_model, model)

        exponent = 0
        for (z_i, data_i, error_i_absolute) in zip(self.zobs, self.data, self.errors):

            # assume a gaussian error
            if error_i_absolute is None or error_i_absolute == 0:
                error_i = 0.
            else:
                error_i = np.random.normal(0, error_i_absolute)

            model_data = interp_model(z_i) + error_i
            delta = model_data - data_i
            exponent += delta ** 2 / 0.05 ** 2

        ndof = len(self.data) - 1
        return exponent / ndof

    def loglike(self, z_model, model):

        return -0.5 * self.chi_square(z_model, model)

    def chi_square(self, z_model, model):

        interp_model = interp1d(z_model, model)

        exponent = 0
        for (z_i, data_i, error_i) in zip(self.zobs, self.data, self.errors):
            model_data = interp_model(z_i)
            delta = model_data - data_i

            dx = delta / error_i
            exponent += dx ** 2

        ndof = len(self.data) - 1
        return exponent / ndof

class JointData(object):

    def __init__(self, data_1, data_2, ignore_1=False, ignore_2=False):

        self.data_1, self.data_2 = data_1, data_2

        self.ignore_1 = ignore_1
        self.ignore_2 = ignore_2

    def summary_statistic(self, z_model_1, z_model_2, model_1, model_2):

        n = 0
        if self.ignore_1:
            pen1 = 0
        else:
            n += 1
            pen1 = self.data_1.summary_statistic(z_model_1, model_1)
        if self.ignore_2:
            pen2 = 0
        else:
            n += 1
            pen2 = self.data_2.summary_statistic(z_model_2, model_2)
        assert n > 0

        return pen1 + pen2

    def chi_square(self, z_model_1, z_model_2, model_1, model_2):

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

    def loglike(self, z_model, model):

        return -0.5 * self.chi_square(z_model, model)

class DistanceCalculator(object):

    def __init__(self, model_domain, data_uncertainties, data_domain,
                 sample_inds_1=None, sample_inds_2=None, ignore_1=False, ignore_2=False):

        self.model_domain = model_domain
        self.data_uncertainties = data_uncertainties
        self.data_domain = data_domain
        self.sample_inds_1 = sample_inds_1
        self.sample_inds_2 = sample_inds_2
        self.ignore1, self.ignore2 = ignore_1, ignore_2

    def _jointdata(self, observed_data):

        data_1 = Data(self.data_domain[0], observed_data[0], self.data_uncertainties[0], self.sample_inds_1)
        data_2 = Data(self.data_domain[1], observed_data[1], self.data_uncertainties[1], self.sample_inds_2)

        joint_data = JointData(data_1, data_2, self.ignore1, self.ignore2)

        return joint_data

    def chi_square(self, observed_data, model_data):

        joint_data = self._jointdata(observed_data)
        (asymmetry, mean_vz) = model_data
        chi2 = joint_data.chi_square(self.model_domain[0], self.model_domain[1], asymmetry, mean_vz)

        return chi2

    def summary_statistic(self, observed_data, model_data):

        joint_data = self._jointdata(observed_data)
        (asymmetry, mean_vz) = model_data
        stat = joint_data.summary_statistic(self.model_domain[0], self.model_domain[1], asymmetry, mean_vz)

        return stat

    def distance(self, observed_data, model_data):

        # PMCABC requires a routine named distance
        return self.summary_statistic(observed_data, model_data)

    def logLike(self, observed_data, model_data):

        return -0.5 * self.chi_square(observed_data, model_data)

    def dist_max(self):
        return 1e+4
