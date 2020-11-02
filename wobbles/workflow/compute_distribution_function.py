from wobbles.disc import Disc
import numpy as np

def compute_df_time_dependent(potential_extension_local, satellite_integration_time_list,
                              satellite_orbit_list, satellite_potential_list, rho_midplane=None, verbose=False):

    """
    Does this seem right?
    :param potential_extension_local:
    :param orbit_integration_time_list:
    :param conversion_to_internal_time:
    :param satellite_orbit_list:
    :param satellite_potential_list:
    :param rho_midplane:
    :return:
    """

    df_list, dj_list, force_list = [], [], []
    disc_instance = Disc(potential_extension_local)

    for satellite_integration_time in satellite_integration_time_list:

        t_end_sat = satellite_integration_time[-1]-satellite_integration_time[0]
        t_eval_orbits = np.linspace(0., t_end_sat, len(satellite_integration_time))

        df,  dj, f = compute_df(disc_instance, satellite_integration_time, satellite_orbit_list, 
                                satellite_potential_list, rho_midplane, t_eval_orbits, verbose)
        df_list.append(df)
        dj_list.append(dj)
        force_list.append(f)

    return df_list, dj_list, force_list

def compute_df(disc, t_eval_satellite, satellite_orbit_list, satellite_potential_list, rho_midplane=None,
               t_eval_orbits=None, verbose=False):

    """
    This function executes a certain workflow sequence: From a the orbit of a passing satellite, compute the
    perturbation to the action in the local phase space, and use it to derive a distribution function for the density/velocity
    in the z direction in the solar neighborhood.

    :param disc: An instance of Disc initialized with a local potential (see documentation in wobbles.wobbles)
    Note that this can be different from the galactic potential used to integrate the orbit of the passing satellite
    :param t_eval_satellite: The time over which to compute the perturbation from the satellite in internal units
    :param satellite_orbit_list: A list of perturbing satellite orbits
    :param satellite_potential_list: A list of satellite potentials corresponding to each orbit
    :param rho_midplane: the midplane density of the disk, needs to be specified for Isothermal potentials. For others it can
    be directly computed from the local potential (see rho_midplane method in potential_extension class)
    :param t_eval_orbits: the times when to evaluate the orbits of test particles in phase space; if None, reverts to the
    same times as those used to sample the satellite orbit
    :param verbose: makes print statements appear

    :return: The distribution function, the perturbation to the action, and the force from the passing satellite
    """

    if t_eval_orbits is None:
        t_eval_orbits = t_eval_satellite

    if verbose:
        print('computing the force from ' + str(len(satellite_orbit_list)) + ' satellite orbits...')

    disc_phase_space_orbits = disc.orbits_in_phase_space(t_eval_orbits)
        
    force = disc.satellite_forces(t_eval_satellite, t_eval_orbits, satellite_orbit_list, satellite_potential_list,
                                  disc_phase_space_orbits, verbose)

    delta_J = disc.action_impulse(force, t_eval_orbits, satellite_orbit_list, satellite_potential_list,
                                 disc_phase_space_orbits)

    dF = disc.distribution_function(delta_J, rho_midplane, verbose)

    return dF, delta_J, force