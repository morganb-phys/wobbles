from wobbles.workflow.compute_distribution_function import compute_df
from wobbles.workflow.integrate_single_orbit import integrate_orbit
from wobbles.workflow.subhalos_and_dwarfs import *
from galpy.potential.mwpotentials import PowerSphericalPotentialwCutoff, MiyamotoNagaiPotential
from wobbles.disc import Disc
from galpy.potential import MovingObjectPotential
import os
import numpy as np

import galpy
from galpy.potential import NFWPotential, HernquistPotential
from wobbles.potential_extension import PotentialExtension
from galpy.orbit.Orbits import Orbit
from scipy.stats.kde import gaussian_kde

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

def single_iteration(samples, tabulated_potential, kde_instance, phase_space_res, **kwargs):

    try:
        potential_local = tabulated_potential.evaluate(samples['rho_nfw'],
                                                   samples['rho_midplane'])

    except:
        # prior sampled out of bounds
        # print('out of bounds: ', samples['nfw_norm'], samples['disk_norm'])
        return None, None, None

    keywords = ['rho_midplane', 'rho_midplane',
                'log_sag_mass_DM', 'sag_mass2light',
                'f_sub', 'log_slope', 'm_host',
                'velocity_dispersion_1', 'velocity_dispersion_2', 'velocity_dispersion_3',
                'component_amplitude_1', 'component_amplitude_2', 'component_amplitude_3',
                 'orbit_ra', 'orbit_dec', 'orbit_z', 'orbit_pm_ra', 'orbit_pm_dec', 'orbit_vlos',
                'gal_norm', 'c8']

    for kw in keywords:
        assert kw in samples.keys(), 'did not find '+str(kw)

    velocity_dispersion = [samples['velocity_dispersion_1']]
    component_amplitude = [samples['component_amplitude_1']]

    if samples['velocity_dispersion_2'] is not None:
        velocity_dispersion.append(samples['velocity_dispersion_2'])
        component_amplitude_2 = 1 - component_amplitude[0]
        component_amplitude.append(component_amplitude_2)

    assert len(velocity_dispersion) == len(component_amplitude)
    assert np.sum(component_amplitude) == 1

    galactic_potential = sample_galactic_potential(samples['gal_norm'])
    z_min_max = 2.
    vmin_max = 100.
    potential_global = PotentialExtension(galactic_potential, z_min_max, vmin_max, phase_space_res,
                                          compute_action_angle=False)
    if 'include_LMC' in kwargs.keys() and kwargs['include_LMC']:
        assert 'log_LMC_mass' in samples.keys()
        LMC_mass = 10 ** samples['log_LMC_mass']
        c = sample_concentration_herquist(LMC_mass, 17.5)
        LMC_potential = HernquistPotential(amp=0.5 * LMC_mass * apu.solMass, a=c * apu.kpc)
        lmc_orbit = Orbit.from_name('LMC')
        lmc_orbit.integrate(time_Gyr, galactic_potential)
        galactic_potential_for_integrations = galactic_potential + MovingObjectPotential(
            lmc_orbit, LMC_potential,
            ro=potential_local.units['ro'],
            vo=potential_local.units['vo'])
    else:
        galactic_potential_for_integrations = galactic_potential

    subhalo_orbit_list, halo_potentials = [], []
    if samples['f_sub'] > 0:
        mlow, mhigh = kwargs['mlow'], kwargs['mhigh']
        print('creating subhalos... ')
        subhalo_orbit_list, halo_potentials = render_subhalos(mlow, mhigh, samples['f_sub'],
                                                              samples['log_slope'], samples['m_host'],
                                                              kde_instance, samples['c8'],
                                                              galactic_potential_for_integrations,
                                                              potential_global,
                                                              time_Gyr, mdef='HERNQUIST', dr_max=10,
                                                              pass_through_disk_limit=3)

    dwarf_orbits, dwarf_galaxy_potentials = [], []
    if 'include_dwarfs' in kwargs.keys() and kwargs['include_dwarfs']:
        o = Orbit.from_name('MW satellite galaxies')
        names = o.name
        kept_names = []
        dwarf_orbits = []

        dwarf_masses, dwarf_names = dwarf_galaxies(names)

        for i, name in enumerate(dwarf_names):
            o = Orbit.from_name(name)
            if o.r() < kwargs['r_min_dwarfs'] and name != 'Sgr':

                o.integrate(time_Gyr, galactic_potential_for_integrations)
                o.turn_physical_off()
                dwarf_orbits += [o]
                kept_names.append(name)

        dwarf_galaxy_potentials, _ = dwarf_galaxies(kept_names, include_no_data=True,
                                                    log_mass_mean=8, log_mass_sigma=0.5)

    ####################################### Integrate orbit of Sag. #################################
    orbit_init_sag = [samples['orbit_ra'] * apu.deg, samples['orbit_dec'] * apu.deg,
                      samples['orbit_z'] * apu.kpc,
                      samples['orbit_pm_ra'] * apu.mas / apu.yr, samples['orbit_pm_dec'] * apu.mas / apu.yr,
                      samples['orbit_vlos'] * apu.km / apu.s]
    sag_orbit_phsical_off = integrate_orbit(orbit_init_sag, galactic_potential_for_integrations, time_Gyr)
    sag_orbit = [sag_orbit_phsical_off]
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

    perturber_orbits = sag_orbit + subhalo_orbit_list + dwarf_orbits
    perturber_potentials = sag_potential + halo_potentials + dwarf_galaxy_potentials

    dF, delta_J, force = compute_df(disc, time_internal_units,
                                    perturber_orbits, perturber_potentials, velocity_dispersion,
                                    component_amplitude, verbose=False)

    asymmetry, mean_vz, density = dF.A, dF.mean_v_relative, dF.density
    # import matplotlib.pyplot as plt
    # plt.plot(asymmetry, color='k')

    return asymmetry, mean_vz, density

# param_prior = []
# param_prior += [['nfw_norm', 'u', [0.15, 0.45], False]]
# param_prior += [['disk_norm', 'u', [0.5, 0.7], False]]
# param_prior += [['log_sag_mass_DM', 'u', [8.7, 11], False]]
# param_prior += [['sag_mass2light', 'g', [50, 5], True]]
# param_prior += [['f_sub', 'f', 0., False]]
# param_prior += [['log_slope', 'f', -1.9, False]]
# param_prior += [['m_host', 'f', 1.2e+12, False]]
# param_prior += [['velocity_dispersion_1', 'u', [15, 25], False]]
# param_prior += [['component_amplitude_1', 'f', 1, False]]
# param_prior += [['velocity_dispersion_2', 'f', None, False]]
# param_prior += [['component_amplitude_2', 'f', None, False]]
# param_prior += [['velocity_dispersion_3', 'f', None, False]]
# param_prior += [['component_amplitude_3', 'f', None, False]]
# param_prior += [['orbit_ra', 'f', 220, False]]
# param_prior += [['orbit_dec', 'f', -50, False]]
# param_prior += [['orbit_z', 'f', 22, False]]
# param_prior += [['orbit_pm_ra', 'f', -3.1, False]]
# param_prior += [['orbit_pm_dec', 'f', -2.8, False]]
# param_prior += [['orbit_vlos', 'f', 140, False]]
# param_prior += [['gal_norm', 'u', [0.7, 1.3], False]]
#
# Nreal = 2000
# VLA_data_path = os.getenv('HOME') + '/Code/external/wobbles/wobbles/workflow/'
# output_folder = './output/forward_model_samples_new/'
# f = open('./saved_potentials/tabulated_MWpot', 'rb')
# tabulated_potential = pickle.load(f)
# f.close()
#
# save_params_list = ['nfw_norm', 'disk_norm', 'log_sag_mass_DM', 'velocity_dispersion_1', 'gal_norm']
# run(1, Nreal, output_folder, VLA_data_path,
#      tabulated_potential, save_params_list, readout_step=50, parameter_priors=param_prior)
