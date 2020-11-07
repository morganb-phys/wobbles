import numpy as np
import galpy
import astropy.units as apu

import clustertools as ctl

def normalization(f, log_slope, m_host, mlow, mhigh):
    denom = mhigh ** (2 + log_slope) - mlow ** (2 + log_slope)
    norm = (2 + log_slope) * m_host * f / denom
    N_halos = norm * (mhigh ** (1 + log_slope) - mlow ** (1 + log_slope)) / (1 + log_slope)
    return np.random.poisson(N_halos)

def sample_mass_function(n, a, ml, mh):
    # samples from a mass function dN/dm propto m^log_slope
    invert_CDF = lambda x: (x * (mh ** (1 + a) - ml ** (1 + a)) + ml ** (
            1 + a)) ** (
                                   (1 + a) ** -1)

    u = np.random.uniform(0, 1, n)

    return invert_CDF(u)


def sample_concentration(m):
    # close to the CDM mass concentration relation
    return 17 * (m / 10 ** 8) ** -0.06

def sample_positions(rmax3d):

    R = np.random.uniform(0, rmax3d)
    r2d = np.random.uniform(0, R)
    u = np.random.rand()
    if u < 0.5:
        sign = -1
    else:
        sign = 1
    z = sign * np.sqrt(R ** 2 - r2d ** 2)
    phi = np.random.uniform(0, 360)
    kpc = apu.kpc
    return R * kpc, z * kpc, phi * apu.deg

def sample_velocities(r, vcirc_mean=220, vcirc_sigma=30):

    u = np.random.rand()
    if u < 0.5:
        sign = -1
    else:
        sign = 1

    rscale = np.sqrt(8/r.value)

    vt = sign * np.random.normal(vcirc_mean, vcirc_sigma)
    vr = np.random.normal(0, 50.) * rscale
    vz = np.random.normal(0, 50.) * rscale
    km_sec = apu.km / apu.s

    return vr * km_sec, vt * km_sec, vz * km_sec

def sample_initial_conditions_ctl(N, potential, r_vir=250):

    subhalo = ctl.setup_cluster('galpy', units='kpckms', origin='galaxy', pot=potential,
                                 N=N, rmax=r_vir)

    o = ctl.initialize_orbits(subhalo)

    return o

def sample_initial_conditions(potential, rmax3d):

    r, z, phi = sample_positions(rmax3d)
    vr, vt, vz = sample_velocities(r)

    return [r, vr, vt, z, vz, phi]

def core_nfw_pdf(r, rs_host, r_core):

    x = r/rs_host
    x_core = r_core/rs_host

    if x > x_core:
        return 1
    else:
        p_core = 1 / (x_core * (1 + x_core) ** 2)
        p = 1 / (x * (1 + x) ** 2)
        ratio = p_core / p
        return ratio

def remove_within_x(r, rs_host, r_core):

    x = r / rs_host
    x_core = r_core / rs_host

    if x > x_core:
        return 1.
    else:
        return 0.

def filter_orbits_NFW(orbits_init, time_in_Gyr, filter_function, function_args):

    inds = []
    t0 = time_in_Gyr[0]
    for i, single_orbit in enumerate(orbits_init):
        x, y, z = single_orbit.x(t0), single_orbit.y(t0), single_orbit.z(t0)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        prob = filter_function(r, *function_args)
        u = np.random.rand()
        accept = prob >= u

        if accept:
            inds.append(i)

    return np.array(inds)

def generate_sample_orbits_ctl(N, potential, time_in_Gyr):

    orbits = sample_initial_conditions_ctl(N, potential)
    orbits.integrate(time_in_Gyr, potential)
    return orbits

def generate_sample_orbits(N, potential, time_in_Gyr, rmax3d=250):

    orbits = []

    for i in range(0, N):

        init = sample_initial_conditions(potential, rmax3d)
        orbit = galpy.orbit.Orbit(vxvv=init)

        orbit.integrate(time_in_Gyr, potential)
        orbits.append(orbit)

    return orbits

def sample_initial_conditions_kde(N, kde):

    (r, vr, vt, z, vz, phi) = kde.resample(N)

    r, vr, vt, z, vz, phi = r[0], vr[0], vt[0], z[0], vz[0], phi[0]

    r *= apu.kpc
    z *= apu.kpc
    vr *= apu.km / apu.s
    vt *= apu.km / apu.s
    vz *= apu.km / apu.s
    phi = phi * 180/np.pi * apu.deg

    return [r, vr, vt, z, vz, phi]

def generate_sample_orbits_kde(N, kde, potential, time_in_Gyr):

    orbits = []

    for i in range(0, N):

        init = sample_initial_conditions_kde(1, kde)
        orbit = galpy.orbit.Orbit(vxvv=init)

        orbit.integrate(time_in_Gyr, potential)
        orbits.append(orbit)

    return orbits

def passed_near_solar_neighorhood(orbit_list, t, potential_extension_global, R_solar=8, dr_max=10, pass_through_disk_limit=2,
                                  tdep=True):

    r_over_r0 = potential_extension_global.R_over_R0_eval

    vc_over_v0 = potential_extension_global.Vc

    freq = vc_over_v0 / r_over_r0

    t_internal_units = potential_extension_global.time_to_internal_time(t.value)

    if tdep:
        x_solar = R_solar * np.cos(freq * t_internal_units)
        y_solar = R_solar * np.sin(freq * t_internal_units)
        z_solar = 0.

    else:
        x_solar = 8.
        y_solar = 0.
        z_solar = 0.

    inds = []

    approach_distance = []

    orbits = []

    for idx, orb in enumerate(orbit_list):

        x_orb, y_orb, z_orb = np.squeeze(orb.x(t)), np.squeeze(orb.y(t)), np.squeeze(orb.z(t))

        dx, dy, dz = x_orb - x_solar, y_orb - y_solar, z_orb - z_solar
        dr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if np.any(dr <= dr_max):

            change_sign_z = np.where(np.sign(z_orb[:-1]) != np.sign(z_orb[1:]))[0] + 1

            # if it never passes through the disk
            if len(change_sign_z) < pass_through_disk_limit:
                inds.append(idx)
                approach_distance.append(min(dr))
                idx_min = np.argmin(dr)
                orbits.append([x_orb[idx_min], y_orb[idx_min], z_orb[idx_min]])
                continue

    sorted_inds = np.argsort(approach_distance)
    orbits_out = []
    for idx in sorted_inds:
        orbits_out.append(orbits[idx])

    return np.array(inds)[sorted_inds], orbits_out
