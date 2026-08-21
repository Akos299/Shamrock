import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import shamrock

shamrock.enable_experimental_features()


# ============================================================
# Matplotlib configuration
# ============================================================

LW = 4
MS = 7
FONTSIZE = 25
TICKWIDTH = 1.5
TICKSIZE = 4

mpl.rcParams.update(
    {
        "axes.titlesize": 1.5 * FONTSIZE,
        "axes.labelsize": 1.5 * FONTSIZE,
        "xtick.major.size": TICKSIZE,
        "ytick.major.size": TICKSIZE,
        "xtick.major.width": TICKWIDTH,
        "ytick.major.width": TICKWIDTH,
        "xtick.minor.size": TICKSIZE,
        "ytick.minor.size": TICKSIZE,
        "xtick.minor.width": TICKWIDTH,
        "ytick.minor.width": TICKWIDTH,
        "lines.linewidth": LW,
        "lines.markersize": MS,
        "lines.markeredgewidth": 1.15,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "font.weight": "bold",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "text.latex.preamble": r"""
            \usepackage{lmodern}
            \usepackage{amsmath}
            \usepackage{amssymb}
        """,
        "axes.unicode_minus": False,
        "pgf.texsystem": "pdflatex",
    }
)


# ============================================================
# Physical / test parameters
# ============================================================

G = 1.0

RHO0 = 2.0
AMP = 10.0

L = 1.0

X0 = 0.5 * L
Y0 = 0.5 * L
Z0 = 0.5 * L

SIGMA = 0.05

# base * sz = actual number of cells per dimension
BASES = [8, 16]
# BASES = [8, 16, 32, 64]

SZ = 2

# Solver parameters
PCG_TOL = 1e-6
BICGSTAB_TOL = 1e-6

BICGSTAB_HAPPY_BREAKDOWN_TOL = 1e-6

SOLVER_MAXITER = 200000

N_BOUND_LENGTH = 3


# ============================================================
# Periodic Gaussian
# ============================================================


def periodic_gaussian(
    x,
    y,
    z,
    sigma=SIGMA,
    amp=AMP,
    x0=X0,
    y0=Y0,
    z0=Z0,
    n=N_BOUND_LENGTH,
):
    res = 0.0

    for nx in range(-n, n + 1):
        for ny in range(-n, n + 1):
            for nz in range(-n, n + 1):
                cx = x - x0 - nx * L
                cy = y - y0 - ny * L
                cz = z - z0 - nz * L

                r2 = cx**2 + cy**2 + cz**2

                res += np.exp(-r2 / (2.0 * sigma**2))

    return amp * res


# ============================================================
# Density
# ============================================================


def rho_function(x, y, z):
    return RHO0 + periodic_gaussian(
        x,
        y,
        z,
        sigma=SIGMA,
        amp=AMP,
        x0=X0,
        y0=Y0,
        z0=Z0,
        n=N_BOUND_LENGTH,
    )


# ============================================================
# Run one Shamrock simulation
# ============================================================


def run_simulation(base, solver):

    N = base * SZ

    print()
    print("=" * 80)
    print(f"Running {solver}: N = {N}^3")
    print("=" * 80)

    # --------------------------------------------------------
    # Shamrock setup
    # --------------------------------------------------------

    ctx = shamrock.Context()

    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(
        context=ctx,
        vector_type="f64_3",
        grid_repr="i64_3",
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    multx = 1
    multy = 1
    multz = 1

    scale_fact = 1.0 / (base * SZ * multx)

    # --------------------------------------------------------
    # Solver configuration
    # --------------------------------------------------------

    cfg = model.gen_default_config()

    cfg.set_scale_factor(scale_fact)

    cfg.set_riemann_solver_hllc()
    cfg.set_eos_gamma(1.4)
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(False)

    if solver == "PCG":
        cfg.set_gravity_mode_cg()

        cfg.set_self_gravity_tol(PCG_TOL)

    elif solver == "BICGSTAB":
        cfg.set_gravity_mode_bicgstab()

        cfg.set_self_gravity_tol(BICGSTAB_TOL)

        cfg.set_self_gravity_happy_breakdown_tol(BICGSTAB_HAPPY_BREAKDOWN_TOL)

    else:
        raise ValueError(f"Unknown solver: {solver}")

    cfg.set_self_gravity_G_values(
        True,
        G,
    )

    cfg.set_self_gravity_Niter_max(SOLVER_MAXITER)

    cfg.set_coupling_gravity_mode_ramses_like()

    model.set_solver_config(cfg)

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    model.init_scheduler(
        int(4_000_000),
        1,
    )

    # --------------------------------------------------------
    # Base grid
    # --------------------------------------------------------

    model.make_base_grid(
        (0, 0, 0),
        (SZ, SZ, SZ),
        (
            base * multx,
            base * multy,
            base * multz,
        ),
    )

    # --------------------------------------------------------
    # Cell-center helper
    # --------------------------------------------------------

    def get_cell_center(rmin, rmax):

        x = 0.5 * (rmin[0] + rmax[0])
        y = 0.5 * (rmin[1] + rmax[1])
        z = 0.5 * (rmin[2] + rmax[2])

        return x, y, z

    # --------------------------------------------------------
    # rho
    # --------------------------------------------------------

    def rho_map(rmin, rmax):

        x, y, z = get_cell_center(
            rmin,
            rmax,
        )

        return rho_function(
            x,
            y,
            z,
        )

    # --------------------------------------------------------
    # Other fields
    # --------------------------------------------------------

    def rhoe_map(rmin, rmax):
        return 0.0

    def rhovel_map(rmin, rmax):

        rho = rho_map(
            rmin,
            rmax,
        )

        return (
            0.0 * rho,
            0.0 * rho,
            0.0 * rho,
        )

    def phi_map(rmin, rmax):
        return 0.0

    # --------------------------------------------------------
    # Register fields
    # --------------------------------------------------------

    model.set_field_value_lambda_f64(
        "rho",
        rho_map,
    )

    model.set_field_value_lambda_f64(
        "rhoetot",
        rhoe_map,
    )

    model.set_field_value_lambda_f64_3(
        "rhovel",
        rhovel_map,
    )

    model.set_field_value_lambda_f64(
        "phi_old",
        phi_map,
    )

    model.set_field_value_lambda_f64(
        "phi",
        phi_map,
    )

    # --------------------------------------------------------
    # Run exactly one evolution
    # --------------------------------------------------------

    dt = 0.0
    t = 0.0

    next_dt = model.evolve_once_override_time(
        t,
        dt,
    )

    t += dt
    dt = next_dt

    # --------------------------------------------------------
    # Number of gravity iterations
    # --------------------------------------------------------

    nb_iter = model.get_self_gravity_nb_iter()

    print(f"{solver}: N = {N}^3 -> {nb_iter} iterations")

    return {
        "N": N,
        "nb_iter": nb_iter,
    }


# ============================================================
# Iteration convergence study
# ============================================================


def convergence_study(
    solvers=("PCG", "BICGSTAB"),
):

    results = {
        solver: {
            "N": [],
            "Niter": [],
        }
        for solver in solvers
    }

    for solver in solvers:
        print()
        print("#" * 80)
        print(f"3D PERIODIC POISSON - {solver}")
        print("#" * 80)

        if solver == "PCG":
            print(f"Tolerance = {PCG_TOL:.1e}")

        elif solver == "BICGSTAB":
            print(f"Tolerance = {BICGSTAB_TOL:.1e}")

            print(f"Happy-breakdown tolerance = {BICGSTAB_HAPPY_BREAKDOWN_TOL:.1e}")

        for base in BASES:
            result = run_simulation(
                base,
                solver,
            )

            results[solver]["N"].append(result["N"])

            results[solver]["Niter"].append(result["nb_iter"])

    # --------------------------------------------------------
    # Convert to numpy arrays
    # --------------------------------------------------------

    for solver in solvers:
        results[solver]["N"] = np.asarray(
            results[solver]["N"],
            dtype=float,
        )

        results[solver]["Niter"] = np.asarray(
            results[solver]["Niter"],
            dtype=float,
        )

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ITERATION CONVERGENCE")
    print("=" * 70)

    header = f"{'N':>10}{'PCG':>15}{'BiCGSTAB':>15}"

    print(header)
    print("-" * 70)

    N_values = results["PCG"]["N"]

    for i, N in enumerate(N_values):
        pcg_iter = results["PCG"]["Niter"][i]
        bicg_iter = results["BICGSTAB"]["Niter"][i]

        print(f"{int(N):>10}{int(pcg_iter):>15}{int(bicg_iter):>15}")

    return results


# ============================================================
# Plot B
#
# Number of solver iterations vs resolution
# ============================================================


def plot_iterations(
    results,
    solvers=("PCG", "BICGSTAB"),
):

    fig, ax = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Measured iteration counts
    # --------------------------------------------------------

    for solver in solvers:
        N = results[solver]["N"]
        iterations = results[solver]["Niter"]

        ax.plot(
            N,
            iterations,
            "D-",
            lw=LW,
            ms=MS,
            label=solver,
        )

    # --------------------------------------------------------
    # O(N) reference
    #
    # Normalize through the first PCG point.
    # --------------------------------------------------------

    N = results["PCG"]["N"]
    pcg_iterations = results["PCG"]["Niter"]

    if len(N) > 0:
        iteration_reference = pcg_iterations[0] * N / N[0]

        ax.plot(
            N,
            iteration_reference,
            "--",
            lw=3,
            label=r"$O(N)$",
        )

    # --------------------------------------------------------
    # Log-log
    # --------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")

    # --------------------------------------------------------
    # X ticks
    # --------------------------------------------------------

    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(N))

    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    ax.set_xticks(N)
    ax.set_xticklabels(
        [str(int(n)) for n in N],
    )

    # --------------------------------------------------------
    # Y ticks
    # --------------------------------------------------------

    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Resolution $N$",
        fontsize=FONTSIZE,
    )

    ax.set_ylabel(
        "Number of iterations",
        fontsize=FONTSIZE,
    )

    # --------------------------------------------------------
    # Tick appearance
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        which="major",
        length=9,
        width=2.5,
        labelsize=FONTSIZE,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=FONTSIZE,
        frameon=True,
    )

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        "poisson_solver_iterations.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        "poisson_solver_iterations.pdf",
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = convergence_study(
        solvers=("PCG", "BICGSTAB"),
    )

    plot_iterations(
        results,
        solvers=("PCG", "BICGSTAB"),
    )


# # ============================================================
# # Configuration
# # ============================================================

# LOG_FILES = {
#     "PCG": "pcg.log",
#     "BICGSTAB": "bicgstab.log",
# }

# OUTPUT = "poisson_solver_iterations.png"

# LW = 4
# MS = 7
# FONTSIZE = 25


# # ============================================================
# # Matplotlib
# # ============================================================

# mpl.rcParams.update(
#     {
#         "axes.titlesize": 1.5 * FONTSIZE,
#         "axes.labelsize": 1.5 * FONTSIZE,

#         "xtick.labelsize": FONTSIZE,
#         "ytick.labelsize": FONTSIZE,

#         "legend.fontsize": FONTSIZE,

#         "lines.linewidth": LW,
#         "lines.markersize": MS,

#         "font.weight": "bold",

#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Latin Modern Roman"],

#         "text.latex.preamble": r"""
#             \usepackage{lmodern}
#             \usepackage{amsmath}
#             \usepackage{amssymb}
#         """,

#         "axes.unicode_minus": False,
#     }
# )


# # ============================================================
# # Parse one log file
# # ============================================================

# def parse_log(filename):
#     """
#     Extract

#         N
#         number of solver iterations

#     from a Shamrock log file.

#     Expected format:

#         Running N = 16^3

#         ...

#         The solution converged after  39 iterations
#     """

#     text = Path(filename).read_text(
#         encoding="utf-8",
#         errors="ignore",
#     )

#     # --------------------------------------------------------
#     # Find every "Running N = ...^3" block
#     # --------------------------------------------------------

#     run_pattern = re.compile(
#         r"Running N = (\d+)\^3"
#     )

#     matches = list(
#         run_pattern.finditer(text)
#     )

#     data = []

#     for i, match in enumerate(matches):

#         N = int(match.group(1))

#         # End of current run
#         if i + 1 < len(matches):
#             end = matches[i + 1].start()
#         else:
#             end = len(text)

#         block = text[
#             match.end():end
#         ]

#         # ----------------------------------------------------
#         # Find convergence messages
#         # ----------------------------------------------------

#         convergence = re.findall(
#             r"The solution converged after\s+(\d+)\s+iterations",
#             block,
#         )

#         if not convergence:
#             print(
#                 f"WARNING: no convergence found for N={N}"
#             )
#             continue

#         # ----------------------------------------------------
#         # Your log can contain the same convergence message
#         # more than once.
#         #
#         # We take the last one in the block.
#         # ----------------------------------------------------

#         niter = int(convergence[-1])

#         data.append(
#             (N, niter)
#         )

#     return np.asarray(
#         data,
#         dtype=float,
#     )


# # ============================================================
# # Load all solvers
# # ============================================================

# results = {}

# for solver, logfile in LOG_FILES.items():

#     data = parse_log(logfile)

#     if len(data) == 0:
#         raise RuntimeError(
#             f"No convergence data found in {logfile}"
#         )

#     results[solver] = {
#         "N": data[:, 0],
#         "Niter": data[:, 1],
#     }


# # ============================================================
# # Print extracted data
# # ============================================================

# print()
# print("=" * 60)
# print("EXTRACTED SOLVER ITERATIONS")
# print("=" * 60)

# for solver, data in results.items():

#     print()
#     print(solver)

#     for N, niter in zip(
#         data["N"],
#         data["Niter"],
#     ):
#         print(
#             f"  N = {int(N):4d} : "
#             f"{int(niter):5d} iterations"
#         )


# # ============================================================
# # Plot B
# # ============================================================

# fig, ax = plt.subplots(
#     figsize=(10, 8),
#     constrained_layout=True,
# )


# # ------------------------------------------------------------
# # Solver curves
# # ------------------------------------------------------------

# for solver, data in results.items():

#     ax.plot(
#         data["N"],
#         data["Niter"],
#         "D-",
#         label=solver,
#     )


# # ============================================================
# # O(N) reference
# # ============================================================

# # Use PCG as normalization if available.
# if "PCG" in results:

#     N_ref = results["PCG"]["N"]
#     iter_ref = results["PCG"]["Niter"]

# else:

#     first_solver = next(iter(results))

#     N_ref = results[first_solver]["N"]
#     iter_ref = results[first_solver]["Niter"]


# reference = (
#     iter_ref[0]
#     * N_ref
#     / N_ref[0]
# )


# ax.plot(
#     N_ref,
#     reference,
#     "--",
#     lw=3,
#     label=r"$O(N)$",
# )


# # ============================================================
# # Log-log axes
# # ============================================================

# ax.set_xscale("log")
# ax.set_yscale("log")


# # ------------------------------------------------------------
# # X ticks
# # ------------------------------------------------------------

# ax.xaxis.set_major_locator(
#     mpl.ticker.FixedLocator(N_ref)
# )

# ax.xaxis.set_minor_locator(
#     mpl.ticker.NullLocator()
# )

# ax.set_xticks(N_ref)

# ax.set_xticklabels(
#     [str(int(N)) for N in N_ref]
# )


# # ------------------------------------------------------------
# # Y ticks
# # ------------------------------------------------------------

# ax.yaxis.set_minor_locator(
#     mpl.ticker.NullLocator()
# )


# # ============================================================
# # Labels
# # ============================================================

# ax.set_xlabel(
#     r"Resolution $N$",
#     fontsize=FONTSIZE,
# )

# ax.set_ylabel(
#     "Number of iterations",
#     fontsize=FONTSIZE,
# )


# # ============================================================
# # Tick appearance
# # ============================================================

# ax.tick_params(
#     axis="both",
#     which="major",
#     length=9,
#     width=2.5,
#     labelsize=FONTSIZE,
# )


# # ============================================================
# # Legend
# # ============================================================

# ax.legend(
#     fontsize=FONTSIZE,
#     frameon=True,
# )


# # ============================================================
# # Frame
# # ============================================================

# for spine in ax.spines.values():
#     spine.set_linewidth(3)


# # ============================================================
# # Save
# # ============================================================

# fig.savefig(
#     OUTPUT,
#     dpi=300,
#     bbox_inches="tight",
# )

# fig.savefig(
#     OUTPUT.replace(".png", ".pdf"),
#     bbox_inches="tight",
# )


# plt.show()
