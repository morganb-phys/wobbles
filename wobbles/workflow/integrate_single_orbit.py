import galpy

def integrate_orbit(orbit_init, potential, time_Gyr, ro=8., vo=220.):

    """
    This function integrates an orbit of a test particle in a potential with initial conditions orbit_init over a time
    time_Gyr

    :param orbit_init: orbit initializtion (see documentation in galpy.Orbit)
    Currently implemented for orbits initialized in a particular coordinte system with radec=True

    :param potential: 3D potential in which to integrate the oribt
    :param time_Gyr: Time in Gyr over which to integrate orbit
    :param ro: length scale for conversion to galpy internal units
    :param vo: velocity scale for conversion to galpy internal units
    :return: Instance of galpy.Oribt
    """

    satellite_orbit = galpy.orbit.Orbit(vxvv=orbit_init, radec=True, ro=ro, vo=vo)  # Initialise orbit instance

    satellite_orbit.integrate(time_Gyr, potential)  # Integrate orbit
    satellite_orbit.turn_physical_off()

    return satellite_orbit