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

        t_end = satellite_integration_time[-1]-satellite_integration_time[0]
        orbit_integration_time= np.linspace(0., t_end, len(satellite_integration_time))
        
        df,  dj, f = compute_df(disc_instance, satellite_integration_time, satellite_orbit_list, 
                                satellite_potential_list, orbit_integration_time,rho_midplane,verbose)
        df_list.append(df)
        dj_list.append(dj)
        force_list.append(f)

    return df_list, dj_list, force_list


def compute_df(disc, satellite_integration_time_units_internal,
               satellite_orbit_list, satellite_potential_list, 
               orbit_integration_time_units_internal= None, rho_midplane=None, verbose=False):

    """
    This function executes a certain workflow sequence: From a the orbit of a passing satellite, compute the
    perturbation to the action in the local phase space, and use it to derive a distribution function for the density/velocity
    in the z direction in the solar neighborhood.

    :param disc: An instance of Disc initialized with a local potential (see documentation in wobbles.wobbles)
    Note that this can be different from the galactic potential used to integrate the orbit of the passing satellite
    :param orbit_integration_time_units_internal: The time over which to compute the perturbation from the satellite in internal units
    :param satellite_orbit_list: A list of perturbing satellite orbits
    :param satellite_potential_list: A list of satellite potentials corresponding to each orbit
    :param rho_midplane: the midplane density of the disk, needs to be specified for Isothermal potentials. For others it can
    be directly computed from the local potential (see rho_midplane method in potential_extension class)
    :param verbose: makes print statements appear

    :return: The distribution function, the perturbation to the action, and the force from the passing satellite
    """

    if verbose:
        print('computing the force from ' + str(len(satellite_orbit_list)) + ' satellite orbits...')

    # Fix this so that the argument that is required for this function does not default to None
    disc_phase_space_orbits = disc.orbits_in_phase_space(orbit_integration_time_units_internal)
        
    force = disc.satellite_forces(satellite_integration_time_units_internal, satellite_orbit_list, satellite_potential_list, 
                                  disc_phase_space_orbits, orbit_integration_time_units_internal,verbose)

    satellite_integration_time_units_internal= satellite_integration_time_units_internal if orbit_integration_time_units_internal is None else orbit_integration_time_units_internal

    delta_J = disc.action_impulse(force, satellite_integration_time_units_internal, satellite_orbit_list, satellite_potential_list,
                                 disc_phase_space_orbits)

    dF = disc.distribution_function(delta_J, rho_midplane, verbose)

    return dF, delta_J, force