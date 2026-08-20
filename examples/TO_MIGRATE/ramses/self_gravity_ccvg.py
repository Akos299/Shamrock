import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LogLocator

import shamrock

shamrock.enable_experimental_features()

# ======================================================
#    Matplotlib config
# ======================================================

lw, ms = 4, 7  # linewidth  #markersize
elw, cs = 0.75, 0.75  # elinewidth and capthick #capsize for errorbar specifically
fontsize = 25
tickwidth, ticksize = 1.5, 4
mpl.rcParams["axes.titlesize"] = fontsize * 1.5
mpl.rcParams["axes.labelsize"] = fontsize * 1.5
mpl.rcParams["xtick.major.size"] = ticksize
mpl.rcParams["ytick.major.size"] = ticksize
mpl.rcParams["xtick.major.width"] = tickwidth
mpl.rcParams["ytick.major.width"] = tickwidth
mpl.rcParams["xtick.minor.size"] = ticksize
mpl.rcParams["ytick.minor.size"] = ticksize
mpl.rcParams["xtick.minor.width"] = tickwidth
mpl.rcParams["ytick.minor.width"] = tickwidth
mpl.rcParams["lines.linewidth"] = lw
mpl.rcParams["lines.markersize"] = ms
mpl.rcParams["lines.markeredgewidth"] = 1.15
mpl.rcParams["lines.dash_joinstyle"] = "bevel"
mpl.rcParams["markers.fillstyle"] = "top"
mpl.rcParams["lines.dashed_pattern"] = 6.4, 1.6, 1, 1.6
mpl.rcParams["xtick.labelsize"] = fontsize
mpl.rcParams["ytick.labelsize"] = fontsize
mpl.rcParams["legend.fontsize"] = fontsize
mpl.rcParams["grid.linewidth"] = 10
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["font.serif"] = "latex"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 20,
        "axes.titlesize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "font.weight": "bold",
    }
)

mpl.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],  # Match lmodern
        # LaTeX preamble to match your class
        "text.latex.preamble": r"""
        \usepackage{lmodern}
        \usepackage{amsmath}
        \usepackage{amssymb}
    """,
        # Optional but recommended
        "axes.unicode_minus": False,
    }
)
mpl.rcParams["pgf.texsystem"] = "pdflatex"


# =======================================================
# Global parameters
# =======================================================

G = 1.0
RHO0 = 2.0
AMP = 10.0
L = 1.0
X0, Y0, Z0 = 0.5 * L, 0.5 * L, 0.5 * L
SIGMA = 0.05
BASES = [8]
# 16, 32]
CG_TOL = 1e-12
CG_MAXITER = 20000
n_bound_lenth = 3
sz = 2


# ========================================================
# Periodic Gaussian
# ========================================================
def periodic_gaussian(x, y, z, sigma=SIGMA, amp=AMP, x0=X0, y0=Y0, z0=Z0, n=n_bound_lenth):

    res = 0.0
    # For sigma = 0.05 note that contribution from n= +/- 3 is negligible.
    for nx in range(-n, n + 1):
        for ny in range(-n, n + 1):
            for nz in range(-n, n + 1):
                cx = x - x0 - nx * L
                cy = y - y0 - ny * L
                cz = z - z0 - nz * L
                r2 = cx**2 + cy**2 + cz**2
                res += np.exp(-r2 / (2.0 * sigma**2))
    return amp * res


# ========================================================
# Density
# ========================================================
def rho_function(x, y, z):
    return RHO0 + periodic_gaussian(
        x, y, z, sigma=SIGMA, amp=AMP, x0=X0, y0=Y0, z0=Z0, n=n_bound_lenth
    )


# ============================================================
# Exact periodic gravitational potential
# ============================================================


def phi_exact_function(x, y, z, sigma=SIGMA, amp=AMP, G=G, x0=X0, y0=Y0, z0=Z0, n=n_bound_lenth):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    res = np.zeros_like(x, dtype=np.float64)

    # --------------------------------------------------------
    # kmax chosen from the Gaussian Fourier decay
    # --------------------------------------------------------

    rel_tol = 1e-13

    kmax = int(np.ceil(np.sqrt(-np.log(rel_tol) / (2.0 * np.pi**2 * sigma**2))))

    # --------------------------------------------------------
    # Fourier sum
    # --------------------------------------------------------

    prefactor = -4.0 * G * np.pi / (L**3) * amp * (2.0 * np.pi) ** 1.5 * sigma**3

    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            for kz in range(-kmax, kmax + 1):
                k2 = kx * kx + ky * ky + kz * kz

                q2 = (2.0 * np.pi / L) ** 2 * k2

                # k = 0 does not have a solution.
                # It corresponds to the mean density.
                if k2 == 0:
                    continue

                fk = np.exp(-(sigma**2) * q2 / 2.0)

                if fk < rel_tol:
                    continue

                phase = 2.0 * np.pi / L * (kx * (x - x0) + ky * (y - y0) + kz * (z - z0))
                res += fk / q2 * np.cos(phase)
    return prefactor * res


# ============================================================
# Error functions
# ============================================================


def l1_diff(f1, f2):
    return np.mean(np.abs(f1 - f2))


def l2_diff(f1, f2):
    return np.sqrt(np.mean((f1 - f2) ** 2))


def linf_diff(f1, f2):
    return np.max(np.abs(f1 - f2))


def l1_norm(f):
    return np.mean(np.abs(f))


def l2_norm(f):
    return np.sqrt(np.mean(f**2))


def linf_norm(f):
    return np.max(np.abs(f))


# ============================================================
# Relative errors
# ============================================================


def compute_errors(phi, phi_exact):

    # --------------------------------------------------------
    # Periodic Poisson determines phi only up to a constant.
    #
    # We remove the mean error before comparison.
    # --------------------------------------------------------

    error = phi - phi_exact

    error -= np.mean(error)

    l1 = np.mean(np.abs(error))
    l2 = np.sqrt(np.mean(error**2))
    linf = np.max(np.abs(error))

    exact_l1 = l1_norm(phi_exact)
    exact_l2 = l2_norm(phi_exact)
    exact_linf = linf_norm(phi_exact)

    return {
        "L1": l1,
        "L2": l2,
        "Linf": linf,
        "L1_rel": l1 / exact_l1,
        "L2_rel": l2 / exact_l2,
        "Linf_rel": linf / exact_linf,
    }


def diagnose_solution(phi, phi_exact):

    phi = np.asarray(phi)
    phi_exact = np.asarray(phi_exact)

    # Remove arbitrary constant
    phi = phi - np.mean(phi)
    phi_exact = phi_exact - np.mean(phi_exact)

    norm_num = np.sqrt(np.mean(phi**2))
    norm_exact = np.sqrt(np.mean(phi_exact**2))

    correlation = np.mean(phi * phi_exact) / (norm_num * norm_exact)

    error_minus = np.sqrt(np.mean((phi - phi_exact) ** 2))

    error_plus = np.sqrt(np.mean((phi + phi_exact) ** 2))

    print()
    print("========== DIAGNOSTIC ==========")
    print(f"||phi_num||       = {norm_num:.12e}")
    print(f"||phi_exact||     = {norm_exact:.12e}")
    print(f"correlation       = {correlation:.12e}")
    print(f"||num - exact||   = {error_minus:.12e}")
    print(f"||num + exact||   = {error_plus:.12e}")
    print("================================")


# ============================================================
# Shamrock simulation config
# ============================================================


def run_simulation(base):

    print()
    print("=" * 80)
    print(f"Running N = {base * sz}^3")
    print("=" * 80)

    # ========================================================
    # Shamrock setup
    # ========================================================

    ctx = shamrock.Context()

    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    # --------------------------------------------------------
    # Grid
    #
    # Your original setup:
    #
    #     sz * base = number of cells
    #
    # We keep base = 8.
    # --------------------------------------------------------

    multx = 1
    multy = 1
    multz = 1

    scale_fact = 1.0 / (base * sz * multx)

    # ========================================================
    # Configuration
    # ========================================================

    cfg = model.gen_default_config()

    cfg.set_scale_factor(scale_fact)

    cfg.set_riemann_solver_hllc()

    cfg.set_eos_gamma(1.4)

    cfg.set_slope_lim_vanleer_sym()

    cfg.set_face_time_interpolation(False)

    # --------------------------------------------------------
    # CG gravity
    # --------------------------------------------------------

    cfg.set_gravity_mode_cg()

    cfg.set_self_gravity_G_values(True, G)

    cfg.set_self_gravity_Niter_max(CG_MAXITER)

    cfg.set_self_gravity_tol(CG_TOL)

    cfg.set_coupling_gravity_mode_ramses_like()

    model.set_solver_config(cfg)

    # ========================================================
    # Scheduler
    # ========================================================

    model.init_scheduler(int(4000000), 1)

    # ========================================================
    # Base grid
    # ========================================================

    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    # ========================================================
    # Shamrock cell-wise initialization
    # ========================================================

    def get_cell_center(rmin, rmax):

        x = 0.5 * (rmin[0] + rmax[0])

        y = 0.5 * (rmin[1] + rmax[1])

        z = 0.5 * (rmin[2] + rmax[2])

        return x, y, z

    # --------------------------------------------------------
    # rho
    # --------------------------------------------------------

    def rho_map(rmin, rmax):

        x, y, z = get_cell_center(rmin, rmax)

        return rho_function(x, y, z)

    # --------------------------------------------------------
    # Total gas energy density
    #
    # rhoetot = rho for the current test
    # --------------------------------------------------------

    def rhoe_map(rmin, rmax):

        # return rho_map(
        #     rmin,
        #     rmax
        # )
        return 0.0

    # --------------------------------------------------------
    # rho velocity
    # --------------------------------------------------------

    def rhovel_map(rmin, rmax):

        rho = rho_map(rmin, rmax)

        return (0.0 * rho, 0.0 * rho, 0.0 * rho)

    # --------------------------------------------------------
    # Initial gravitational potential
    #
    # phi_0 = 0
    # --------------------------------------------------------

    def phi_map(rmin, rmax):

        return 0.0

    # ========================================================
    # Register fields
    # ========================================================

    model.set_field_value_lambda_f64("rho", rho_map)

    model.set_field_value_lambda_f64("rhoetot", rhoe_map)

    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    model.set_field_value_lambda_f64("phi_old", phi_map)

    model.set_field_value_lambda_f64("phi", phi_map)

    # ========================================================
    # Run
    # ========================================================

    dt = 0.0
    t = 0.0

    Max_iter_run = 1

    for k in range(Max_iter_run):
        next_dt = model.evolve_once_override_time(t, dt)

        t += dt

        dt = next_dt

    nb_iter_sg = model.get_self_gravity_nb_iter()

    # ========================================================
    # Collect Shamrock data
    # ========================================================

    dic = ctx.collect_data()

    print("len(cell_min) =", len(dic["cell_min"]))
    print("len(cell_max) =", len(dic["cell_max"]))
    print("len(rho)      =", len(dic["rho"]))
    print("len(phi)      =", len(dic["phi"]))
    print("len(phi_old)  =", len(dic["phi_old"]))

    print("max |phi|     = ", np.max(np.abs(dic["phi"])))

    print("max |phi_old| = ", np.max(np.abs(dic["phi_old"])))

    print("L2 |phi|       = ", np.sqrt(np.mean(np.asarray(dic["phi"]) ** 2)))

    print("L2 |phi_old|   = ", np.sqrt(np.mean(np.asarray(dic["phi_old"]) ** 2)))

    # ========================================================
    # Coordinates
    # ========================================================

    cmin = dic["cell_min"]
    cmax = dic["cell_max"]

    X = []
    Y = []
    Z = []

    rho = []
    phi = []

    for i in range(len(cmin)):
        m = cmin[i]
        M = cmax[i]

        # ----------------------------------------------------
        # get_cell_coords returns the physical coordinates
        # of the subcells
        # ----------------------------------------------------
        for j in range(8):
            a, b = model.get_cell_coords((tuple(m), tuple(M)), j)

            xmin, ymin, zmin = a
            xmax, ymax, zmax = b

            x = 0.5 * (xmin + xmax)

            y = 0.5 * (ymin + ymax)

            z = 0.5 * (zmin + zmax)

            idx = 8 * i + j

            X.append(x)
            Y.append(y)
            Z.append(z)

            # -----------------------------------------------
            # Numerical fields
            # -----------------------------------------------

            rho.append(dic["rho"][idx])

            # phi.append(
            #     dic["phi_old"][i]
            # )

            phi.append(dic["phi"][idx])

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    rho = np.asarray(rho)
    # ========================================================
    # Diagnostic rho
    # ========================================================
    phi = np.asarray(phi)

    rho_exact = rho_function(X, Y, Z)

    rho_error = rho - rho_exact

    print()
    print("========== RHO DIAGNOSTIC ==========")

    print("rho numerical min/max =", np.min(rho), np.max(rho))

    print("rho exact min/max     =", np.min(rho_exact), np.max(rho_exact))

    print("max |rho-rho_exact|   =", np.max(np.abs(rho_error)))

    print("L2 |rho-rho_exact|    =", np.sqrt(np.mean(rho_error**2)))

    print("mean rho numerical    =", np.mean(rho))

    print("mean rho exact        =", np.mean(rho_exact))

    print("====================================")

    # ========================================================
    # Exact solution at Shamrock cell centers
    # ========================================================

    phi_exact = phi_exact_function(X, Y, Z)
    # ========================================================
    # Diagnostic phi
    # ========================================================
    diagnose_solution(phi, phi_exact)

    # ========================================================
    # Errors
    # ========================================================

    errors = compute_errors(phi, phi_exact)

    # ========================================================
    # Return everything useful
    # ========================================================
    return {
        "N": base * sz,
        "X": X,
        "Y": Y,
        "Z": Z,
        "rho": rho,
        "phi_0": np.zeros_like(phi),
        "phi": phi,
        "phi_exact": phi_exact,
        "errors": errors,
        "nb_iter": nb_iter_sg,
    }


# ============================================================
# Convergence study
# ============================================================
def convergence_study():

    results = []

    print()
    print("=" * 100)
    print("3D PERIODIC POISSON - SHAMROCK CG")
    print("=" * 100)
    print(f"Fixed sigma = {SIGMA}")
    print(f"CG tolerance = {CG_TOL}")
    print()

    for B in BASES:
        result = run_simulation(B)

        results.append(result)

        errors = result["errors"]

        print()
        print(
            f"N = {B * sz:4d}  "
            f"L1 = {errors['L1_rel']:.6e}  "
            f"L2 = {errors['L2_rel']:.6e}  "
            f"Linf = {errors['Linf_rel']:.6e}  "
            f"Niter = {result['nb_iter']}"
        )

    # ========================================================
    # Convergence orders
    # ========================================================

    N_values = np.array([r["N"] for r in results], dtype=float)

    L1_errors = np.array([r["errors"]["L1_rel"] for r in results])

    L2_errors = np.array([r["errors"]["L2_rel"] for r in results])

    Linf_errors = np.array([r["errors"]["Linf_rel"] for r in results])

    Niter_values = np.array([r["nb_iter"] for r in results])

    def get_order(errors):

        orders = np.full(len(errors), np.nan)

        for i in range(1, len(errors)):
            orders[i] = np.log2(errors[i - 1] / errors[i])

        return orders

    order_L1 = get_order(L1_errors)
    order_L2 = get_order(L2_errors)
    order_Linf = get_order(Linf_errors)

    # ========================================================
    # Print convergence table
    # ========================================================

    print()
    print("=" * 100)
    print("CONVERGENCE")
    print("=" * 100)

    print(f"{'N':>8}{'L1':>16}{'Order':>10}{'L2':>16}{'Order':>10}{'Linf':>16}{'Order':>10}")

    print("-" * 100)

    for i, N in enumerate(N_values):
        print(
            f"{int(N):8d}"
            f"{L1_errors[i]:16.6e}"
            f"{order_L1[i]:10.4f}"
            f"{L2_errors[i]:16.6e}"
            f"{order_L2[i]:10.4f}"
            f"{Linf_errors[i]:16.6e}"
            f"{order_Linf[i]:10.4f}"
        )

    # ========================================================
    # Convergence plots
    # ========================================================

    fig, axs = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    # plt.subplots_adjust(wspace=0.25, hspace=0.3, top=0.94, bottom=0.7, left=0.6, right=0.94)

    # ========================================================
    # Common tick configuration
    # ========================================================

    x_ticks = N_values
    # ========================================================
    # LEFT : error convergence
    # ========================================================

    ax = axs[0]

    ax.plot(N_values, L1_errors, "o-", lw=4, ms=7, label=r"$L_1$")

    ax.plot(N_values, L2_errors, "s-", lw=4, ms=7, label=r"$L_2$")

    ax.plot(N_values, Linf_errors, "^-", lw=4, ms=7, label=r"$L_\infty$")

    # --------------------------------------------------------
    # O(h^2) reference
    # --------------------------------------------------------

    reference = L2_errors[0] * (N_values[0] / N_values) ** 2

    axs[0].plot(N_values, reference, "--", lw=3, label=r"$O(h^2)$")

    # --------------------------------------------------------
    # Log scales
    # --------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")

    # --------------------------------------------------------
    # ONLY major ticks
    # --------------------------------------------------------

    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(N_values))

    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(r"Resolution $N$", fontsize=fontsize)

    ax.set_ylabel(
        r"$\frac{\left\|\Phi_{\rm num}-\Phi_{\rm ana}\right\|}"
        r"{\left\|\Phi_{\rm ana}\right\|}$",
        fontsize=1.3 * fontsize,
    )

    # --------------------------------------------------------
    # Tick labels
    # --------------------------------------------------------

    ax.set_xticklabels([str(int(N)) for N in N_values], fontsize=fontsize)

    ax.tick_params(axis="both", which="major", length=9, width=2.5, labelsize=fontsize)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(fontsize=fontsize, frameon=True)

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # ========================================================
    # RIGHT : CG iterations
    # ========================================================

    ax = axs[1]

    # --------------------------------------------------------
    # Reference O(N)
    # --------------------------------------------------------

    iteration_reference = Niter_values[0] * N_values / N_values[0]

    # --------------------------------------------------------
    # Measured iterations
    # --------------------------------------------------------

    ax.plot(N_values, Niter_values, "D-", lw=4, ms=7, label="CG")

    ax.plot(N_values, iteration_reference, "k--", lw=3, label=r"$O(N)$")

    # --------------------------------------------------------
    # Log-log
    # --------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")

    # --------------------------------------------------------
    # ONLY major ticks
    # --------------------------------------------------------

    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(N_values))

    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    # Choose readable iteration ticks
    iteration_ticks = [50, 100, 150, 200, 250]

    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(iteration_ticks))

    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(r"Resolution $N$", fontsize=fontsize)

    ax.set_ylabel("Number of CG iterations", fontsize=fontsize)

    # --------------------------------------------------------
    # Tick labels
    # --------------------------------------------------------

    ax.set_xticklabels([str(int(N)) for N in N_values], fontsize=fontsize)

    ax.set_yticklabels([str(v) for v in iteration_ticks], fontsize=fontsize)

    ax.tick_params(axis="both", which="major", length=9, width=2.5, labelsize=fontsize)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(fontsize=fontsize, frameon=True)

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # ========================================================
    # Save
    # ========================================================

    fig.savefig("poisson_convergence_fixed_sigma.png", dpi=300, bbox_inches="tight")

    plt.show()

    # ========================================================
    # Return results
    # ========================================================

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = convergence_study()
