import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration
# ============================================================

LOGFILE = Path("/Users/lsewanou/SHAMROCK_DIR_AMR_POISSON/Shamrock/build_new/out_comp_1.txt")

TOL = 1e-6

# Output files
PLOT_A = "plot_A_residual_history.pdf"
PLOT_B = "plot_B_iterations_vs_resolution.pdf"


# ============================================================
# Regex
# ============================================================

# Example:
# Running PCG: N = 16^3
RUN_RE = re.compile(r"Running\s+(PCG|BICGSTAB):\s*N\s*=\s*(\d+)\^3")

# Example:
# [PCG] k = 12 ... ||r_k||_2 / ||b_rhs||_2 = 0.0845
PCG_RES_RE = re.compile(
    r"\[PCG\].*?k\s*=\s*(\d+).*?"
    r"\|\|r_k\|\|_2\s*/\s*\|\|b_rhs\|\|_2\s*=\s*"
    r"([0-9eE.+\-]+)"
)

# Example:
# [BICGSTAB] k = 12 ||r_k||_2 / ||b_rhs||_2 = 0.0197
BICGSTAB_RES_RE = re.compile(
    r"\[BICGSTAB\].*?k\s*=\s*(\d+).*?"
    r"\|\|r_k\|\|_2\s*/\s*\|\|b_rhs\|\|_2\s*=\s*"
    r"([0-9eE.+\-]+)"
)

# Final iteration summary:
#
# PCG:    N = 16^3 -> 39 iterations
# BICGSTAB: N = 16^3 -> 30 iterations
ITER_RE = re.compile(r"(PCG|BICGSTAB):\s*N\s*=\s*(\d+)\^3\s*->\s*(\d+)\s*iterations")


# ============================================================
# Read logfile
# ============================================================

text = LOGFILE.read_text(errors="replace")


# ============================================================
# Parse residual histories
# ============================================================

# Structure:
#
# histories["PCG"][16] = [(0, 1.0), (1, ...), ...]
# histories["BICGSTAB"][32] = [...]
#
histories = {
    "PCG": {},
    "BICGSTAB": {},
}


current_method = None
current_N = None


for line in text.splitlines():
    # --------------------------------------------------------
    # Detect beginning of a solver run
    # --------------------------------------------------------
    run_match = RUN_RE.search(line)

    if run_match:
        current_method = run_match.group(1)
        current_N = int(run_match.group(2))

        histories[current_method].setdefault(current_N, [])

        continue

    # --------------------------------------------------------
    # PCG residual
    # --------------------------------------------------------
    if current_method == "PCG":
        match = PCG_RES_RE.search(line)

        if match:
            k = int(match.group(1))
            residual = float(match.group(2))

            histories["PCG"][current_N].append((k, residual))

            continue

    # --------------------------------------------------------
    # BiCGSTAB residual
    # --------------------------------------------------------
    if current_method == "BICGSTAB":
        match = BICGSTAB_RES_RE.search(line)

        if match:
            k = int(match.group(1))
            residual = float(match.group(2))

            histories["BICGSTAB"][current_N].append((k, residual))

            continue


# ============================================================
# Remove duplicated convergence histories
# ============================================================


def extract_longest_history(data):
    """
    The logfile currently contains the same solver history twice.

    Example:

        k=0
        k=1
        ...
        k=39

        k=0
        k=1
        ...
        k=39

    We split whenever k goes backwards and keep the longest
    contiguous history.
    """

    if not data:
        return []

    sequences = []
    current = [data[0]]

    for item in data[1:]:
        k_previous = current[-1][0]
        k_current = item[0]

        # A new history starts when k goes backwards.
        if k_current <= k_previous:
            sequences.append(current)
            current = [item]

        else:
            current.append(item)

    sequences.append(current)

    # Keep the longest sequence
    longest = max(sequences, key=len)

    # Also remove accidental duplicate k values
    result = {}

    for k, residual in longest:
        result[k] = residual

    return sorted(result.items())


for method in histories:
    for N in histories[method]:
        histories[method][N] = extract_longest_history(histories[method][N])


# ============================================================
# Parse final iteration summary
# ============================================================

iterations = {
    "PCG": {},
    "BICGSTAB": {},
}


for match in ITER_RE.finditer(text):
    method = match.group(1)
    N = int(match.group(2))
    niter = int(match.group(3))

    iterations[method][N] = niter


# ============================================================
# Print what was extracted
# ============================================================

print("\n============================================================")
print("Residual histories")
print("============================================================")

for method in ["PCG", "BICGSTAB"]:
    print(f"\n{method}")

    for N in sorted(histories[method]):
        data = histories[method][N]

        if not data:
            print(f"  N={N}: NO DATA")
            continue

        k = np.array([x[0] for x in data])
        r = np.array([x[1] for x in data])

        print(f"  N={N:4d}: {len(k):3d} points, k={k[0]} -> {k[-1]}, r_final={r[-1]:.6e}")


print("\n============================================================")
print("Final iteration counts")
print("============================================================")

for N in sorted(set(iterations["PCG"]) | set(iterations["BICGSTAB"])):
    print(
        f"N={N:4d} : "
        f"PCG={iterations['PCG'].get(N, 'N/A'):>4}   "
        f"BiCGSTAB={iterations['BICGSTAB'].get(N, 'N/A'):>4}"
    )


# ============================================================
# Plot A
#
# Residual convergence history
#
# One panel per resolution.
# PCG and BiCGSTAB are compared directly.
# ============================================================

resolutions = sorted(set(histories["PCG"]) | set(histories["BICGSTAB"]))

fig, axes = plt.subplots(
    1,
    len(resolutions),
    figsize=(6.5 * len(resolutions), 5.0),
    squeeze=False,
)

axes = axes[0]

for ax, N in zip(axes, resolutions):
    # --------------------------------------------------------
    # PCG
    # --------------------------------------------------------

    if N in histories["PCG"]:
        data = histories["PCG"][N]

        k = np.array([x[0] for x in data])
        r = np.array([x[1] for x in data])

        ax.semilogy(
            k,
            r,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="PCG",
        )

    # --------------------------------------------------------
    # BiCGSTAB
    # --------------------------------------------------------

    if N in histories["BICGSTAB"]:
        data = histories["BICGSTAB"][N]

        k = np.array([x[0] for x in data])
        r = np.array([x[1] for x in data])

        ax.semilogy(
            k,
            r,
            marker="s",
            markersize=3,
            linewidth=1.8,
            label="BiCGSTAB",
        )

    # --------------------------------------------------------
    # Convergence tolerance
    # --------------------------------------------------------

    ax.axhline(
        TOL,
        linestyle="--",
        linewidth=1.2,
        label=r"$10^{-6}$ tolerance",
    )

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|r_k\|_2 / \|b\|_2$")

    ax.set_title(rf"$N={N}^3$")

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.5,
    )

    ax.legend()

    ax.set_ylim(
        bottom=1e-7,
        top=2,
    )


fig.suptitle(
    "PCG vs BiCGSTAB convergence history",
    fontsize=15,
)

fig.tight_layout()

fig.savefig(
    PLOT_A,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Plot B
#
# Number of iterations vs resolution
# ============================================================

Ns = sorted(set(iterations["PCG"]) | set(iterations["BICGSTAB"]))

pcg_iters = [iterations["PCG"].get(N, np.nan) for N in Ns]

bicgstab_iters = [iterations["BICGSTAB"].get(N, np.nan) for N in Ns]


fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    Ns,
    pcg_iters,
    marker="o",
    linewidth=1.8,
    markersize=6,
    label="PCG",
)

ax.plot(
    Ns,
    bicgstab_iters,
    marker="s",
    linewidth=1.8,
    markersize=6,
    label="BiCGSTAB",
)

ax.set_xlabel(r"Resolution $N$")

ax.set_ylabel("Number of iterations")

ax.set_title("Solver iteration count vs resolution")

ax.grid(
    True,
    linestyle=":",
    alpha=0.5,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PLOT_B,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Final summary
# ============================================================

print("\n============================================================")
print("Generated plots")
print("============================================================")

print(f"Plot A : {PLOT_A}")
print(f"Plot B : {PLOT_B}")
