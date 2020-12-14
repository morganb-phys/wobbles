import numpy as np
import galpy
import astropy.units as apu
from galpy.potential import NFWPotential, HernquistPotential


def dwarf_galaxies(names, include_no_data=True,
                   log_mass_mean=8., log_mass_sigma=0.5):

    # these from Errani et al. 2018
    dwarfs = {'Fornax': 8.9, 'LeoI': 9.3, 'Sculptor': 9.3, 'LeoII': 8.5, 'Sextans': 8.5, 'Carina': 8.6,
               'UrsaMinor': 8.9, 'Draco': 9.5, 'CanesVenaticiI': 8.5, 'CraterII': 7., 'LeoT': 9.8, 'Hercules': 7.1,
               'BootesI': 6.4, 'LeoIV': 7.2, 'UrsaMajorI': 8.5, 'UrsaMajorII': 9.1,
              'CanesVenaticiII': 8.7, 'ComaBerenices': 8.6,
               'BootesII': 10.4, 'Willman1': 10.4, 'Segue2': 9., 'Segue1': 9.8, 'LeoV': 7.5}

    dwarf_masses = []
    dwarf_names = []

    for name in names:
        if name in dwarfs.keys():
            dwarf_masses.append(dwarfs[name])
            dwarf_names.append(name)
        else:
            if include_no_data:
                logm = np.random.normal(log_mass_mean, log_mass_sigma)
                dwarf_masses.append(logm)
                dwarf_names.append(name)
            else:
                print('excluing dwarf '+name)

    dwarf_potentials = []
    for logm in dwarf_masses:

        m = 10 ** logm
        c = sample_concentration_herquist(m, 17.5)
        pot = HernquistPotential(amp=0.5 * m * apu.solMass, a=c * apu.kpc)
        dwarf_potentials.append(pot)

    return dwarf_potentials, dwarf_names

def render_subhalos(mlow, mhigh, f_sub, log_slope, m_host, via_lactea_kde,
                    c8, galactic_potential, global_potential_extension,
                    time_Gyr, mdef='HERNQUIST', dr_max=8, pass_through_disk_limit=3):

    N_halos = normalization(f_sub, log_slope, m_host, mlow, mhigh)

    sample_orbits_0 = generate_sample_orbits_kde(
        N_halos, via_lactea_kde, galactic_potential, time_Gyr)
    # print('generated ' + str(N_halos) + ' halos... ')
    rs_host, r_core = 25, 25.
    args, func = (rs_host, r_core), core_nfw_pdf
    inds_keep = filter_orbits_NFW(sample_orbits_0, time_Gyr, func, args)
    sample_orbits_1 = [sample_orbits_0[idx] for idx in inds_keep]

    nearby_orbits_1_inds, _ = passed_near_solar_neighorhood(sample_orbits_1, time_Gyr,
                                       global_potential_extension, R_solar=8,
                                       dr_max=dr_max, pass_through_disk_limit=pass_through_disk_limit,
                                                            tdep=True)

    nearby_orbits_1 = [sample_orbits_0[idx] for idx in nearby_orbits_1_inds]
    n_nearby_1 = len(nearby_orbits_1)

    halo_masses_1 = sample_mass_function(n_nearby_1, log_slope, mlow, mhigh)

    halo_potentials = []

    if mdef == 'HERNQUIST':
        concentrations = sample_concentration_herquist(halo_masses_1, c8)

        for m, c in zip(halo_masses_1, concentrations):
            pot = HernquistPotential(amp=0.5 * m * apu.solMass, a=c * apu.kpc)
            halo_potentials.append(pot)

    elif mdef == 'NFW':
        concentrations = sample_concentration_nfw(halo_masses_1, c8)
        for m, c in zip(halo_masses_1, concentrations):
            pot = NFWPotential(mvir=m / 10 ** 12, conc=c)
            halo_potentials.append(pot)

    subhalo_orbit_list = []
    for orb in nearby_orbits_1:
        orb.turn_physical_off()
        subhalo_orbit_list.append(orb)

    return subhalo_orbit_list, halo_potentials


def fsub_transform(f_sub_6_10, log_slope, mL, mH):

    mH_ref = 10**10
    mL_ref = 10**6
    mean_6_10 = mH_ref ** (2 + log_slope) - mL_ref ** (2 + log_slope)
    mean_mL_mH = mH ** (2 + log_slope) - mL ** (2 + log_slope)
    return f_sub_6_10 * mean_mL_mH / mean_6_10

def normalization(f, log_slope, m_host, mlow, mhigh):

    """
    Returns the number of subhalos between mass mlow and mhigh sammpled from a
    power law mass function with logarithmic slope log_slope.
    The normalization is determined by f, which is the mass fraction in substructure
    between 10^6 and 10^10.

    :param f: mass function normalization, the mass fraction in subhalos between 10^6 and 10^10
    :param log_slope: logarithmic slope of the mass function
    :param m_host: host halo mass
    :param mlow: low mass end of rendered subhalos
    :param mhigh: high mass end of rendered subhalos
    :return: number of halos
    """
    mH_ref = 10 ** 10
    mL_ref = 10 ** 6

    first_moment = mhigh ** (1 + log_slope) - mlow ** (1 + log_slope)
    second_moment = mH_ref ** (2 + log_slope) - mL_ref ** (2 + log_slope)
    a0 = f * (2 + log_slope) * m_host / second_moment
    n = a0 * first_moment / (1 + log_slope)

    return np.random.poisson(n)

def sample_mass_function(n, a, ml, mh):
    # samples from a mass function dN/dm propto m^log_slope
    invert_CDF = lambda x: (x * (mh ** (1 + a) - ml ** (1 + a)) + ml ** (
            1 + a)) ** (
                                   (1 + a) ** -1)

    u = np.random.uniform(0, 1, n)

    return invert_CDF(u)

def sample_concentration_nfw(m, norm=17.5):
    # close to the CDM mass concentration relation
    return norm * (m / 10 ** 8) ** -0.06

def sample_concentration_herquist(m, norm=17.5):

    a = 1.05 * (m / 10 ** 8) ** 0.5
    rescale = 17.5/norm
    return rescale * a

def sample_hernquist_a(m):

    return 1.05 * (m/10**8) ** 0.5

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

def sample_initial_conditions_kde(kde):

    (r, vr, vt, z, vz, phi) = kde.resample(1)

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

        init = sample_initial_conditions_kde(kde)
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
