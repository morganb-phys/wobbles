from wobbles.wobbles import Disc
import numpy as np

def compute_df_time_dependent(potential_extension_local, orbit_integration_time_Gyr_list, conversion_to_internal_time,
               satellite_orbit_list, satellite_potential_list, rho_midplane=None):

    """
    Does this seem right?
    :param potential_extension_local:
    :param orbit_integration_time_Gyr_list:
    :param conversion_to_internal_time:
    :param satellite_orbit_list:
    :param satellite_potential_list:
    :param rho_midplane:
    :return:
    """

    df_list, dj_list, force_list = [], [], []
    disc_instance = Disc(potential_extension_local)

    for orbit_integration_time in orbit_integration_time_Gyr_list:

        df,  dj, f = compute_df(disc_instance, orbit_integration_time, conversion_to_internal_time,
                         satellite_orbit_list, satellite_potential_list, rho_midplane)
        df_list.append(df)
        dj_list.append(dj)
        force_list.append(f)

    return df_list, dj_list, force_list


def compute_df(disc, orbit_integration_time_Gyr, conversion_to_internal_time,
               satellite_orbit_list, satellite_potential_list, rho_midplane=None):

    """
    This function executes a certain workflow sequence: From a the orbit of a passing satellite, compute the
    perturbation to the action in the local phase space, and use it to derive a distribution function for the density/velocity
    in the z direction in the solar neighborhood.

    :param disc: An instance of Disc initialized with a local potential (see documentation in wobbles.wobbles)
    Note that this can be different from the galactic potential used to integrate the orbit of the passing satellite
    :param orbit_integration_time_Gyr: The time over which to compute the perturbation from the satellite
    :param conversion_to_internal_time: a conversion factor between Gyr and galpy internal time units
    :param satellite_orbit_list: A list of perturbing satellite orbits
    :param satellite_potential_list: A list of satellite potentials corresponding to each orbit
    :param rho_midplane: the midplane density of the disk, needs to be specified for Isothermal potentials. For others it can
    be directly computed from the local potential (see rho_midplane method in potential_extension class)

    :return: The distribution function, the perturbation to the action, and the force from the passing satellite
    """

    t_end = 0
    t_start = orbit_integration_time_Gyr[-1].value - orbit_integration_time_Gyr[0].value
    n_time_steps = len(satellite_orbit_list[0].time())

    time = np.linspace(t_end, t_start * conversion_to_internal_time, n_time_steps)

    force = disc.satellite_forces(time, satellite_orbit_list, satellite_potential_list)

    delta_J = disc.action_impulse(force, time, satellite_orbit_list, satellite_potential_list)

    dF = disc.distribution_function(delta_J, rho_midplane)

    return dF, delta_J, force