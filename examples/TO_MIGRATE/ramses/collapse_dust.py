import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import shamrock

# ============================================================
# SHAMROCK initialization
# ============================================================

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# ============================================================
# Utility functions
# ============================================================


def get_mass(R, rho):
    return rho * (4.0 * np.pi / 3.0) * R**3


def get_omega(G, beta, M, R):
    return np.sqrt(3.0 * beta * G * M / R**3)


# ============================================================
# MRN dust distribution
# ============================================================


def dust_bins(epsilon0=0.01, Smin=5e-9, Smax=250e-9, N=10, m=3.5):
    """
    Compute logarithmic dust bins and dust fraction in each bin.

    Parameters
    ----------
    epsilon0 : float
        Total initial dust fraction:
            epsilon0 = rho_dust / (rho_gas + rho_dust)

    Smin : float
        Minimum grain size [m].

    Smax : float
        Maximum grain size [m].

    N : int
        Number of dust bins.

    m : float
        Power-law exponent:
            dn/ds ~ s^(-m)

        Standard MRN:
            m = 3.5

    Returns
    -------
    Sk : ndarray
        Bin edges, N+1 values.

    sk : ndarray
        Representative grain size of each bin, N values.

    epsilon_k : ndarray
        Dust fraction associated with each bin.
        Sum(epsilon_k) = epsilon0.
    """

    # --------------------------------------------------------
    # 1. Logarithmic bin edges
    # --------------------------------------------------------

    k = np.arange(N + 1)

    Sk = Smin * (Smax / Smin) ** (k / N)

    # --------------------------------------------------------
    # 2. Representative grain size
    #    Geometric mean
    # --------------------------------------------------------

    sk = np.sqrt(Sk[:-1] * Sk[1:])

    # --------------------------------------------------------
    # 3. Dust fraction in each bin
    # --------------------------------------------------------

    if np.isclose(m, 4.0):
        # Special case m = 4
        epsilon_k = epsilon0 * np.log(Sk[1:] / Sk[:-1]) / np.log(Smax / Smin)

    else:
        exponent = 4.0 - m

        denominator = Smax**exponent - Smin**exponent

        epsilon_k = epsilon0 * (Sk[1:] ** exponent - Sk[:-1] ** exponent) / denominator

    return Sk, sk, epsilon_k


# ============================================================
# Unit system
# ============================================================

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)

codeu = shamrock.UnitSystem(
    unit_time=1,  # [s]
    unit_length=1,  # [m]
    unit_mass=1,  # [kg]
)

ucte = shamrock.Constants(codeu)

G = ucte.G()
kb = ucte.kb()
m_H = ucte.proton_mass()


# ============================================================
# Physical parameters
# ============================================================

T0 = 10.747  # [K]

R0 = 7.07e16 * 1e-2  # [cm -> m]

rho0 = 1.38e-18 * 1e3  # [g/cm^3 -> kg/m^3]

M0 = get_mass(R0, rho0)

mu = 2.3  # molecular gas


# ------------------------------------------------------------
# Total dust fraction
#
# epsilon0 = rho_dust / (rho_gas + rho_dust)
# ------------------------------------------------------------

epsilon0 = 0.00990099


# ------------------------------------------------------------
# MRN parameters
# ------------------------------------------------------------

Smin = 5e-9  # 5 nm
Smax = 250e-9  # 250 nm

Nbins = 10

m = 3.5  # Standard MRN


# ------------------------------------------------------------
# Grain material density
# ------------------------------------------------------------

rho_grain = 3.0e3  # [kg/m^3]


# ------------------------------------------------------------
# Total initial dust density
# ------------------------------------------------------------

rhod0 = (epsilon0 / (1.0 - epsilon0)) * rho0


# ------------------------------------------------------------
# Single grain sizes
# ------------------------------------------------------------

s_grain_5nm = 5e-9  # 5 nm

s_grain_1um = 1e-6  # 1 micron

s_grain_10um = 10e-6  # 10 micron

s_grain_100um = 100e-6  # 100 micron


# ============================================================
# Print basic parameters
# ============================================================

print()
print("============================================================")
print("Physical parameters")
print("============================================================")

print(f"proton mass      = {m_H}")
print(f"rho_gas initial  = {rho0:.8e} kg/m^3")
print(f"rho_dust initial = {rhod0:.8e} kg/m^3")
print(f"epsilon0         = {epsilon0:.8e}")
print(f"grain density    = {rho_grain:.8e} kg/m^3")

print()
print(f"5 nm             = {s_grain_5nm:.8e} m")
print(f"1 micron         = {s_grain_1um:.8e} m")
print(f"10 micron        = {s_grain_10um:.8e} m")
print(f"100 micron       = {s_grain_100um:.8e} m")


# ============================================================
# Energies / characteristic quantities
# ============================================================

E_th0 = (3.0 * M0 * kb * T0) / (2.0 * mu * m_H)

E_grav0 = (-3.0 * G * M0**2) / (5.0 * R0)

alpha0 = E_th0 / np.abs(E_grav0)


# ------------------------------------------------------------
# Free-fall time
# ------------------------------------------------------------

t_ff = np.sqrt((3.0 * np.pi) / (32.0 * G * rho0))


# ------------------------------------------------------------
# Sound speed
# ------------------------------------------------------------

cs_sqr = (kb * T0) / (mu * m_H)


# ------------------------------------------------------------
# Jeans length
# ------------------------------------------------------------

lamb_J = np.sqrt((cs_sqr * np.pi) / (G * rho0))


# ------------------------------------------------------------
# Numerical parameters
# ------------------------------------------------------------

N_J = 16  # cells per Jeans length

L0 = 4.0 * R0

min_reso = (L0 * N_J) / lamb_J

gamma = 5.0 / 3.0


# ------------------------------------------------------------
# Critical density
# ------------------------------------------------------------

rho_c = 3.7e-13 * 1e3  # [g/cm^3 -> kg/m^3]


# ============================================================
# Print characteristic quantities
# ============================================================

print()
print("============================================================")
print("Characteristic quantities")
print("============================================================")

print(f"kb               = {kb}")
print(f"G                = {G}")
print(f"Jeans length     = {lamb_J:.8e} m")
print(f"sound speed      = {np.sqrt(cs_sqr):.8e} m/s")
print(f"alpha            = {alpha0:.8e}")
print(f"free-fall time   = {t_ff / (3600 * 24 * 365):.8e} years")
print(f"minimum reso     = {min_reso:.8e}")


# ============================================================
# Compute MRN distribution
# ============================================================

Sk, sk, epsilon_k = dust_bins(epsilon0=epsilon0, Smin=Smin, Smax=Smax, N=Nbins, m=m)


# ============================================================
# Print MRN distribution
# ============================================================

print()
print("============================================================")
print("MRN dust distribution")
print("============================================================")

print(
    f"{'Bin':>5} "
    f"{'S_k [nm]':>15} "
    f"{'S_k+1 [nm]':>15} "
    f"{'s_k [nm]':>15} "
    f"{'epsilon_k':>18} "
    f"{'fraction':>15}"
)

print("-" * 95)

for k in range(Nbins):
    fraction = epsilon_k[k] / epsilon0

    print(
        f"{k:5d} "
        f"{Sk[k] * 1e9:15.6f} "
        f"{Sk[k + 1] * 1e9:15.6f} "
        f"{sk[k] * 1e9:15.6f} "
        f"{epsilon_k[k]:18.10e} "
        f"{fraction:15.8f}"
    )

print("-" * 95)

print(f"Total epsilon0   = {epsilon0:.10e}")
print(f"Sum epsilon_k    = {np.sum(epsilon_k):.10e}")

print()
print("Check:")
print("relative error   =", abs(np.sum(epsilon_k) - epsilon0) / epsilon0)


# ============================================================
# Main simulation
# ============================================================


def run_sim(
    beta=0.04,
    A=0.1,
    with_rotation=False,
    with_fragmentation=False,
    with_mrn_distribution=False,
    rho_grain=rho_grain,
    s_grain=None,
    run_name="run",
):

    print()
    print()
    print("============================================================")
    print(f"Starting simulation: {run_name}")
    print("============================================================")

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    omega_0 = get_omega(G, beta, M0, R0)

    print(f"omega_0 = {omega_0:.8e} [1/s]")

    # ========================================================
    # SHAMROCK setup
    # ========================================================

    shamrock.enable_experimental_features()

    ctx = shamrock.Context()

    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    # ========================================================
    # Grid
    # ========================================================

    multx = 1
    multy = 1
    multz = 1

    max_amr_lev = 18

    sz = 2 << max_amr_lev

    base = 32

    # ========================================================
    # Solver configuration
    # ========================================================

    cfg = model.gen_default_config()

    scale_fact = L0 / (sz * base * multx)

    cfg.set_scale_factor(scale_fact)

    cfg.set_Csafe(0.3)

    cfg.set_eos_gamma(gamma)

    if not with_mrn_distribution:
        cfg.set_dust_mode_hb(1)
    elif with_mrn_distribution:
        cfg.set_dust_mode_hb(Nbins)

    cfg.set_drag_mode_irk1(True)

    cfg.set_slope_lim_minmod()

    cfg.set_face_time_interpolation(True)

    # ========================================================
    # Grain configuration
    # ========================================================

    if not with_mrn_distribution:
        # ----------------------------------------------------
        # Mono-grain simulation
        # ----------------------------------------------------

        if s_grain is None:
            raise ValueError("s_grain must be specified for mono-grain runs.")

        print()
        print("Dust model: mono-grain")
        print(f"grain size = {s_grain * 1e9:.6f} nm")
        print(f"grain density = {rho_grain:.6e} kg/m^3")

        # Intrinsic grain material density
        cfg.set_grains_intrinsic_density_values(rho_grain)

        # Grain size
        cfg.set_grains_sizes_values(s_grain)

    else:
        # ----------------------------------------------------
        # MRN multi-grain simulation
        # ----------------------------------------------------

        print()
        print("Dust model: MRN")
        print(f"Number of bins = {Nbins}")
        print(f"m = {m}")
        print(f"Smin = {Smin * 1e9:.6f} nm")
        print(f"Smax = {Smax * 1e9:.6f} nm")

        for k in range(Nbins):
            # Same material density for all grains
            cfg.set_grains_intrinsic_density_values(rho_grain)

            # Representative grain size
            cfg.set_grains_sizes_values(sk[k])

            print(f"  bin {k:2d}: s = {sk[k] * 1e9:.6f} nm, epsilon = {epsilon_k[k]:.8e}")

    # ========================================================
    # Gravity
    # ========================================================

    cfg.set_gravity_mode_cg()

    # Alternative:
    # cfg.set_gravity_mode_bicgstab()

    cfg.set_riemann_solver_hllc()

    cfg.set_self_gravity_G_values(True, G)

    cfg.set_self_gravity_Niter_max(10000)

    cfg.set_self_gravity_tol(1e-6)

    cfg.set_coupling_gravity_mode_ramses_like()

    cfg.set_amr_mode_jeans_length_based(N_jeans=N_J, T_init=T0)

    # ========================================================
    # Apply solver configuration
    # ========================================================

    model.set_solver_config(cfg)

    model.init_scheduler(int(5000000), 1)

    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    # ========================================================
    # Cell center
    # ========================================================

    def cell_center(rmin, rmax):

        x = 0.5 * (rmin[0] + rmax[0]) - 0.5 * L0

        y = 0.5 * (rmin[1] + rmax[1]) - 0.5 * L0

        z = 0.5 * (rmin[2] + rmax[2]) - 0.5 * L0

        return x, y, z

    # ========================================================
    # Gas density
    # ========================================================

    def rho_map(rmin, rmax):

        x, y, z = cell_center(rmin, rmax)

        r = np.sqrt(x**2 + y**2 + z**2)

        # Outside cloud
        rho_ret = rho0 / 100.0

        # Inside cloud
        if r < R0:
            rho_ret = rho0

            if with_fragmentation:
                phi = np.arctan2(y, x)

                rho_ret *= 1.0 + A * np.cos(2.0 * phi)

        return rho_ret

    # ========================================================
    # Gas momentum
    # ========================================================

    def rhovel_map(rmin, rmax):

        x, y, z = cell_center(rmin, rmax)

        r = np.sqrt(x**2 + y**2 + z**2)

        rho = rho_map(rmin, rmax)

        if with_rotation and r < R0:
            vx = -omega_0 * y
            vy = omega_0 * x
            vz = 0.0

            return (rho * vx, rho * vy, rho * vz)

        return (0.0, 0.0, 0.0)

    # ========================================================
    # Gas total energy
    # ========================================================

    def rhoe_map(rmin, rmax):

        rho = rho_map(rmin, rmax)

        rhov = rhovel_map(rmin, rmax)

        Ekin = 0.5 * (rhov[0] ** 2 + rhov[1] ** 2 + rhov[2] ** 2) / rho

        x = rho / rho_c

        P = cs_sqr * rho * (1.0 + x ** (2.0 / 3.0))

        Eint = P / (gamma - 1.0)

        return Ekin + Eint

    # ========================================================
    # Dust density map factory
    # ========================================================

    def make_rho_dust_map(dust_fraction):
        """
        Create the density map for one dust species.

        dust_fraction:
            fraction of total dust mass carried
            by this grain population.

        For mono-grain:
            dust_fraction = 1

        For MRN:
            dust_fraction = epsilon_k / epsilon0
        """

        def rho_d_map(rmin, rmax):

            # Local gas density
            rho_g = rho_map(rmin, rmax)

            # Local total dust density
            #
            # epsilon0 =
            # rho_d / (rho_g + rho_d)
            #
            # therefore:
            #
            # rho_d =
            # epsilon0/(1-epsilon0) * rho_g
            #
            rho_d_total = (epsilon0 / (1.0 - epsilon0)) * rho_g

            # Dust density in this species
            return dust_fraction * rho_d_total

        return rho_d_map

    # ========================================================
    # Dust velocity / momentum
    # ========================================================

    def rhovel_d_map(rmin, rmax):
        return (0.0, 0.0, 0.0)

    # ========================================================
    # Initialize gas fields
    # ========================================================

    model.set_field_value_lambda_f64("rho", rho_map)

    model.set_field_value_lambda_f64("rhoetot", rhoe_map)

    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # ========================================================
    # Initialize dust fields
    # ========================================================

    if not with_mrn_distribution:
        # ----------------------------------------------------
        # ONE dust species
        # ----------------------------------------------------

        dust_fraction = 1.0

        rho_d_map = make_rho_dust_map(dust_fraction)

        model.set_field_value_lambda_f64("rho_dust", rho_d_map, 0)

        model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_map, 0)

        print()
        print(f"Total initial dust density = {rhod0:.8e} kg/m^3")

        print("Dust species = 1")

        print("Dust fraction = 1.0")

    else:
        # ----------------------------------------------------
        # MRN: Nbins dust species
        # ----------------------------------------------------

        print()
        print("Initializing MRN dust fields...")

        for k in range(Nbins):
            # epsilon_k is the dust fraction
            # associated with bin k.
            #
            # epsilon_k / epsilon0
            # gives the fraction of TOTAL dust mass
            # carried by this bin.

            dust_fraction = epsilon_k[k] / epsilon0

            rho_d_map = make_rho_dust_map(dust_fraction)

            model.set_field_value_lambda_f64("rho_dust", rho_d_map, k)

            model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_d_map, k)

            print(f"bin {k:2d}: dust fraction = {dust_fraction:.10e}, s = {sk[k] * 1e9:.6f} nm")

    # ========================================================
    # Time integration
    # ========================================================

    tmax = 1.5 * t_ff

    t = 0.0

    dt = 0.0

    freq = 10

    # ========================================================
    # Time loop
    # ========================================================

    for i in range(int(1e7)):
        next_dt = model.evolve_once_override_time(t, dt)

        t += dt

        dt = next_dt

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        if i % freq == 0:
            filename = f"{run_name}_{t / t_ff:5f}.vtk"

            model.dump_vtk(filename)

        # ----------------------------------------------------
        # Adjust last timestep
        # ----------------------------------------------------

        if tmax < t + next_dt:
            dt = tmax - t

        # ----------------------------------------------------
        # End simulation
        # ----------------------------------------------------

        if t >= tmax:
            filename = f"{run_name}_{t / t_ff:5f}.vtk"

            model.dump_vtk(filename)

            print()
            print(f"Simulation {run_name} finished at t/t_ff = {t / t_ff:.6f}")

            break


# ============================================================
# RUN 1
# Mono-grain: 5 nm
# ============================================================

run_sim(
    with_rotation=False,
    with_fragmentation=False,
    with_mrn_distribution=False,
    s_grain=s_grain_5nm,
    rho_grain=rho_grain,
    run_name="collapse_5nm",
)


# # ============================================================
# # RUN 2
# # Mono-grain: 1 micron
# # ============================================================

# run_sim(
#     with_rotation=False,
#     with_fragmentation=False,
#     with_mrn_distribution=False,

#     s_grain=s_grain_1um,
#     rho_grain=rho_grain,

#     run_name="collapse_1um"
# )


# # ============================================================
# # RUN 3
# # Mono-grain: 10 micron
# # ============================================================

# run_sim(
#     with_rotation=False,
#     with_fragmentation=False,
#     with_mrn_distribution=False,

#     s_grain=s_grain_10um,
#     rho_grain=rho_grain,

#     run_name="collapse_10um"
# )


# # ============================================================
# # RUN 4
# # Mono-grain: 100 micron
# # ============================================================

# run_sim(
#     with_rotation=False,
#     with_fragmentation=False,
#     with_mrn_distribution=False,

#     s_grain=s_grain_100um,
#     rho_grain=rho_grain,

#     run_name="collapse_100um"
# )


# # ============================================================
# # RUN 5
# # MRN distribution: 10 bins
# # ============================================================

# run_sim(
#     with_rotation=False,
#     with_fragmentation=False,
#     with_mrn_distribution=True,

#     rho_grain=rho_grain,

#     run_name="collapse_MRN_10bins"
# )


# # ============================================================
# # Optional rotating runs
# # ============================================================

# # run_sim(
# #     with_rotation=True,
# #     with_fragmentation=False,
# #     with_mrn_distribution=False,
# #     s_grain=s_grain_1um,
# #     rho_grain=rho_grain,
# #     run_name="rotating_1um"
# # )


# # run_sim(
# #     with_rotation=True,
# #     with_fragmentation=True,
# #     with_mrn_distribution=False,
# #     s_grain=s_grain_1um,
# #     rho_grain=rho_grain,
# #     run_name="rotating_fragmentation_1um"
# # )
