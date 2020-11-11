from wobbles.workflow.compute_distribution_function import compute_df
from wobbles.workflow.integrate_single_orbit import integrate_orbit
from wobbles.workflow.generate_perturbing_subhalos import *
from galpy.potential.mwpotentials import PowerSphericalPotentialwCutoff, MiyamotoNagaiPotential
from wobbles.disc import Disc
import os
import numpy as np

import galpy
from galpy.potential import NFWPotential, HernquistPotential
from wobbles.potential_extension import PotentialExtension

import pickle
from scipy.stats.kde import gaussian_kde
import sys
#
# f = open('./saved_potentials/MW14pot_100', "rb")
# potential_global = pickle.load(f)
# f.close()
#
# galactic_potential = potential_global.galactic_potential

t_orbit = -1.64  # Gyr
N_tsteps = 1200
time_Gyr = np.linspace(0., t_orbit, N_tsteps) * apu.Gyr

def VLA_simulation_phasespaceKDE(path_to_file):
    vla_subhalo_phase_space = np.loadtxt(path_to_file + 'vl2_halos_scaled.dat')
    kde = gaussian_kde(vla_subhalo_phase_space, bw_method=0.1)
    return kde

def sample_galactic_potential(sigma_norm):

    sigma_scale = 1.

    mwpot = [PowerSphericalPotentialwCutoff(normalize=sigma_norm * 0.05, alpha=1.8,
                                    rc=sigma_scale * 1.9 / 8.),
            MiyamotoNagaiPotential(a=sigma_scale * 3. / 8., b= sigma_scale * 0.28 / 8., normalize=sigma_norm * 0.6),
            NFWPotential(a=sigma_scale * 2., normalize=sigma_norm * 0.35)]

    return mwpot

def sample_sag_orbit():

    alpha_0, delta_0 = 220, -50
    d_alpha = np.random.normal(0, 0.001)
    d_delta = np.random.normal(0, 0.001)
    alpha, delta = alpha_0 + d_alpha, delta_0 + d_delta

    z_0 = 22
    delta_z = np.random.normal(0, 0.00005)
    z = z_0 + delta_z

    mu_alpha = -3.1 + np.random.normal(0, 0.000001)
    mu_delta = -2.8 + np.random.normal(0, 0.000001)

    vr_0 = 140
    delta_vr = np.random.normal(0, 0.00000005)
    vr = vr_0 + delta_vr

    orbit_init_sag = [alpha * apu.deg, delta * apu.deg, z * apu.kpc,
                      mu_alpha * apu.mas / apu.yr, mu_delta * apu.mas / apu.yr,
                      vr * apu.km / apu.s]  # Initial conditions of the satellite

    return orbit_init_sag

def run(run_index, Nreal, output_folder_name, VLA_data_path,
        tabulated_potential, save_params_list, readout_step=100):

    # 'nfw_norm', 'disk_norm',
    #  'log_sag_mass_DM', 'sag_mass2light',
    #  'f_sub',
    #  'velocity_dispersion', 'compoent_amplitude',
    #  'orbit_ra', 'orbit_dec', 'orbit_z', 'orbit_pm_ra', 'orbit_pm_dec', 'orbit_vlos'

    init_arrays = True
    count = 0

    for j in range(0, Nreal):
        print(str(j)+' out of '+str(Nreal))
        nfw_norm = np.random.uniform(0.15, 0.45)
        disk_norm = np.random.uniform(0.5, 0.7)
        log_sag_mass_DM = np.random.uniform(9, 11)
        sag_mass2light = abs(np.random.normal(50, 5))
        f_sub = 0.
        log_slope = -1.9
        m_host = 1.3 * 10 ** 12
        velocity_dispersion_1, amp_1 = np.random.uniform(15, 25), 1.
        velocity_dispersion_2, amp_2 = None, None
        velocity_dispersion_3, amp_3 = None, None
        orbit_ra = 220
        orbit_dec = -50
        orbit_z = 22
        orbit_pm_ra, orbit_pm_dec = -3.1, -2.8
        orbit_vlos = np.random.normal(140, 5)
        gal_norm = np.random.uniform(0.7, 1.3)

        samples = {'nfw_norm': nfw_norm, 'disk_norm': disk_norm, 'log_sag_mass_DM': log_sag_mass_DM,
                   'sag_mass2light': sag_mass2light, 'f_sub': f_sub, 'log_slope': log_slope, 'm_host': m_host,
                   'velocity_dispersion_1': velocity_dispersion_1, 'velocity_dispersion_2': velocity_dispersion_2,
                  'velocity_dispersion_3': velocity_dispersion_3, 'component_amplitude_1': amp_1,
                   'component_amplitude_2': amp_2, 'component_amplitude_3': amp_3,
                   'orbit_ra': orbit_ra, 'orbit_dec': orbit_dec,
                   'orbit_z': orbit_z, 'orbit_pm_ra': orbit_pm_ra, 'orbit_pm_dec': orbit_pm_dec,
                   'orbit_vlos': orbit_vlos, 'gal_norm': gal_norm}

        A, vz = _run(VLA_data_path, tabulated_potential, samples)
        for param in save_params_list:
            assert param in samples.keys()
        new_params_sampled = [samples[param] for param in save_params_list]
        new_params_sampled = np.array(new_params_sampled)

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
        if count < readout_step:
            readout = False
        else:
            readout = True

        if readout:

            init_arrays = True
            count = 0
            with open(output_folder_name + 'asymmetry_' + str(run_index) + '.txt', 'a') as f:
                string_to_write = ''
                for row in range(0, asymmetry.shape[0]):
                    for ai in asymmetry[row,:]:
                        string_to_write += str(np.round(ai, 5)) + ' '
                    string_to_write += '\n'
                f.write(string_to_write)

            with open(output_folder_name + 'meanvz_' + str(run_index) + '.txt', 'a') as f:
                string_to_write = ''
                for row in range(0, mean_vz.shape[0]):
                    for vzi in mean_vz[row, :]:
                        string_to_write += str(np.round(vzi, 5)) + ' '
                    string_to_write += '\n'
                f.write(string_to_write)

            with open(output_folder_name + 'params_' + str(run_index) + '.txt', 'a') as f:
                string_to_write = ''
                for row in range(0, params_sampled.shape[0]):
                    for param_val in params_sampled[row, :]:
                        string_to_write += str(np.round(param_val, 5)) + ' '
                    string_to_write += '\n'
                f.write(string_to_write)

def _run(VLA_data_path, tabulated_potential, samples):

    # f is the mass fraction contained in halos between 10^6 and 10^10, CDM prediction is a few percent

    keywords = ['nfw_norm', 'disk_norm',
                'log_sag_mass_DM', 'sag_mass2light',
                'f_sub', 'log_slope', 'm_host',
                'velocity_dispersion_1', 'velocity_dispersion_2', 'velocity_dispersion_3',
                'component_amplitude_1', 'component_amplitude_2', 'component_amplitude_3',
                 'orbit_ra', 'orbit_dec', 'orbit_z', 'orbit_pm_ra', 'orbit_pm_dec', 'orbit_vlos',
                'gal_norm']

    for kw in keywords:
        assert kw in samples.keys(), 'did not find '+str(kw)

    velocity_dispersion = [samples['velocity_dispersion_1']]
    component_amplitude = [samples['component_amplitude_1']]
    if samples['velocity_dispersion_2'] is not None:
        velocity_dispersion.append(samples['velocity_dispersion_2'])
        component_amplitude.append(samples['component_amplitude_2'])
        if samples['velocity_dispersion_3'] is not None:
            velocity_dispersion.append(samples['velocity_dispersion_3'])
            component_amplitude.append(samples['component_amplitude_3'])
    assert len(velocity_dispersion) == len(component_amplitude)

    potential_local = tabulated_potential.evaluate(samples['nfw_norm'],
                                                   samples['disk_norm'])

    galactic_potential = sample_galactic_potential(samples['gal_norm'])

    potential_global = PotentialExtension(galactic_potential, 2, 120, 100,
                                          compute_action_angle=False)

    kde = VLA_simulation_phasespaceKDE(VLA_data_path)

    mlow, mhigh = 5 * 10 ** 6, 10 ** 9
    N_halos = normalization(samples['f_sub'], samples['log_slope'], samples['m_host'], mlow, mhigh)

    ####################################### Generate subhalo orbits #######################################
    sample_orbits_0 = generate_sample_orbits_kde(N_halos, kde, galactic_potential, time_Gyr)
    #print('generated ' + str(N_halos) + ' halos... ')
    rs_host, r_core = 30, 30.
    args, func = (rs_host, r_core), core_nfw_pdf
    inds_keep = filter_orbits_NFW(sample_orbits_0, time_Gyr, func, args)
    sample_orbits_1 = [sample_orbits_0[idx] for idx in inds_keep]
    dr_max = 8  # kpc
    nearby_orbits_1_inds, _ = passed_near_solar_neighorhood(sample_orbits_1, time_Gyr, potential_global, R_solar=8,
                                                     dr_max=dr_max, pass_through_disk_limit=3, tdep=True)
    nearby_orbits_1 = [sample_orbits_0[idx] for idx in nearby_orbits_1_inds]
    n_nearby_1 = len(nearby_orbits_1)
    #print('kept ' + str(n_nearby_1) + ' halos... ')
    ############################################################################################################

    ####################################### Set subhalo properties #######################################
    halo_masses_1 = sample_mass_function(n_nearby_1, samples['log_slope'], mlow, mhigh)
    halo_concentrations_1 = sample_concentration(halo_masses_1)
    halo_potentials_1 = []
    for m, c in zip(halo_masses_1, halo_concentrations_1):
        halo_potentials_1.append(NFWPotential(mvir=m / 10 ** 12, conc=c))
    #####################################################################################################

    ####################################### Integrate orbit of Sag. #################################
    orbit_init_sag = [samples['orbit_ra'] * apu.deg, samples['orbit_dec'] * apu.deg,
                      samples['orbit_z'] * apu.kpc,
                      samples['orbit_pm_ra'] * apu.mas / apu.yr, samples['orbit_pm_dec'] * apu.mas / apu.yr,
                      samples['orbit_vlos'] * apu.km / apu.s]
    sag_orbit_phsical_off = integrate_orbit(orbit_init_sag, galactic_potential, time_Gyr)
    sag_orbit = [sag_orbit_phsical_off]
    halo_orbit_list_physical_off_1 = []
    for orb in nearby_orbits_1:
        orb.turn_physical_off()
        halo_orbit_list_physical_off_1.append(orb)
    #######################################################################################################

    ####################################### Set Sag. properties ########################################
    m_sag_dm = 10 ** samples['log_sag_mass_DM']
    m_sag_stellar = m_sag_dm/samples['sag_mass2light']

    a_sag = 3 * (m_sag_dm/10**10) ** 1./3
    a_stellar = 1.5 * (m_sag_stellar/10**9) ** 1./3

    sag_potential_1 = galpy.potential.HernquistPotential(amp=m_sag_dm * apu.M_sun, a=a_sag * apu.kpc)
    sag_potential_2 = galpy.potential.HernquistPotential(amp=m_sag_stellar * apu.M_sun, a=a_stellar * apu.kpc)
    sag_potential = [sag_potential_1 + sag_potential_2]
    galpy.potential.turn_physical_off(sag_potential)
    #######################################################################################################

    ############################# Compute distribution function ############################################
    disc = Disc(potential_local, potential_global)
    time_internal_units = sag_orbit_phsical_off.time()

    perturber_orbits = sag_orbit + halo_orbit_list_physical_off_1
    perturber_potentials = sag_potential + halo_potentials_1
    dF, delta_J, force = compute_df(disc, time_internal_units,
                                    perturber_orbits, perturber_potentials, velocity_dispersion,
                                    component_amplitude, verbose=False)

    asymmetry, mean_vz = dF.A, dF.mean_v_relative
    return asymmetry, mean_vz

#
Nreal = 2000
VLA_data_path = os.getenv('HOME') + '/Code/external/wobbles/wobbles/workflow/'
output_folder = './output/forward_model_samples_new/'
f = open('./saved_potentials/tabulated_MWpot', 'rb')
tabulated_potential = pickle.load(f)
f.close()

save_params_list = ['nfw_norm', 'disk_norm', 'log_sag_mass_DM', 'velocity_dispersion_1']

    #print(str(Nreal - iter) + ' remaining...')

run(int(sys.argv[1]), Nreal, output_folder, VLA_data_path,
    tabulated_potential, save_params_list, readout_step=50)
