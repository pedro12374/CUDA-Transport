"""
Quantifiers for escape-basin analysis: basin entropy, basin boundary
entropy, the Wada-property merging test, basin area fractions, the MSD
diffusion exponent, and a grid-based estimate of the basin boundary's
fractal dimension via the uncertainty exponent method.

All of these operate on data the CUDA solvers already compute and save to
HDF5 (EscapeBasin_*, MSD_*/MSD_sample_times) -- nothing here needs the GPU.
The one exception in spirit is the fractal-dimension method, which is
*conceptually* about perturbing trajectories and re-simulating; the
implementation here instead reuses the already-computed basin grid's own
spacing as the smallest perturbation scale (see uncertainty_exponent_
fractal_dimension's docstring for the tradeoff this makes).

References:
  - Basin entropy / boundary entropy / merging (Wada) test:
    Daza, Wagemakers, Sanjuan & Yorke, "Basin entropy: a new tool to
    analyze uncertainty in dynamical systems", Sci. Rep. 6, 31416 (2016).
  - Uncertainty exponent / fractal dimension:
    Grebogi, McDonald, Ott & Yorke, "Final state sensitivity: An obstruction
    to predictability", Phys. Lett. A 99, 415 (1983).
"""

import itertools
import numpy as np
from scipy.optimize import curve_fit


# ==============================================================================
# == Area fractions
# ==============================================================================

def basin_area_fractions(basin_grid):
    """{basin_id: fraction_of_cells}, for a basin-ID array of any shape."""
    ids, counts = np.unique(basin_grid, return_counts=True)
    total = basin_grid.size
    return {int(i): float(c) / total for i, c in zip(ids, counts)}


# ==============================================================================
# == Basin entropy / basin boundary entropy (Daza et al. 2016)
# ==============================================================================

def _box_entropy(box):
    """Shannon entropy (natural log) of the color distribution within one box."""
    _, counts = np.unique(box, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _iter_boxes(grid, box_size):
    """Non-overlapping box_size x box_size sub-arrays; drops partial edge boxes."""
    nx, ny = grid.shape
    nbx, nby = nx // box_size, ny // box_size
    for i in range(nbx):
        for j in range(nby):
            yield grid[i * box_size:(i + 1) * box_size, j * box_size:(j + 1) * box_size]


def basin_entropy(basin_grid, box_size=5):
    """
    Basin entropy Sb: average Shannon entropy of the basin-color
    distribution over non-overlapping box_size x box_size boxes. 2D only --
    squeeze out any singleton dimensions first (e.g. a fixed px0/py0 axis).
    """
    boxes = list(_iter_boxes(basin_grid, box_size))
    if not boxes:
        raise ValueError(f"box_size={box_size} too large for grid shape {basin_grid.shape}")
    return float(np.mean([_box_entropy(b) for b in boxes]))


def basin_boundary_entropy(basin_grid, box_size=5):
    """
    Basin boundary entropy Sbb: same as basin_entropy, but averaged only
    over boxes containing more than one basin color (i.e. straddling a
    boundary). Returns (Sbb, num_boundary_boxes, num_total_boxes).
    """
    boxes = list(_iter_boxes(basin_grid, box_size))
    boundary = [b for b in boxes if len(np.unique(b)) > 1]
    if not boundary:
        return 0.0, 0, len(boxes)
    sbb = float(np.mean([_box_entropy(b) for b in boundary]))
    return sbb, len(boundary), len(boxes)


# ==============================================================================
# == Wada-property merging test
# ==============================================================================

def wada_merging_test(basin_grid, box_size=5, num_pairs=None, seed=0):
    """
    Merges random pairs of basin colors and recomputes the boundary
    entropy. If the boundary has the Wada property (every boundary point
    borders *every* basin), merging any pair shouldn't significantly change
    Sbb -- since the merged region's boundary is, statistically, the same
    boundary as before. A pair whose merge visibly drops Sbb suggests that
    pair doesn't fully share its boundary with the rest.

    num_pairs: test only a random subset of all C(num_basins,2) pairs
    (default: all of them).

    Returns a list of dicts: {"pair": (a,b), "sbb_before", "sbb_after", "ratio"}.
    ratio close to 1 for all pairs is evidence for the Wada property; a
    ratio well below 1 for some pair is evidence against it (for that pair).
    """
    ids = [int(i) for i in np.unique(basin_grid)]
    if len(ids) < 3:
        raise ValueError("Need at least 3 distinct basins for a meaningful Wada test")

    sbb0, _, _ = basin_boundary_entropy(basin_grid, box_size)

    pairs = list(itertools.combinations(ids, 2))
    rng = np.random.default_rng(seed)
    if num_pairs is not None and num_pairs < len(pairs):
        idx = rng.choice(len(pairs), size=num_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    results = []
    for a, b in pairs:
        merged = basin_grid.copy()
        merged[merged == b] = a
        sbb_ab, _, _ = basin_boundary_entropy(merged, box_size)
        results.append({
            "pair": (a, b),
            "sbb_before": sbb0,
            "sbb_after": sbb_ab,
            "ratio": (sbb_ab / sbb0) if sbb0 > 0 else float("nan"),
        })
    return results


# ==============================================================================
# == MSD diffusion exponent
# ==============================================================================

def fit_diffusion_exponent(t, msd, fit_start=1):
    """
    Fits MSD(t) = D * t^alpha by nonlinear least squares (skipping the
    first `fit_start` samples, typically t=0 where MSD is trivially 0).
    alpha ~ 1: normal diffusion. alpha < 1: sub-diffusion. alpha > 1:
    super-diffusion (alpha ~ 2: ballistic).

    Returns (D, alpha, D_stderr, alpha_stderr).
    """
    def power_law(t, D, alpha):
        return D * (t ** alpha)

    t_fit = np.asarray(t)[fit_start:]
    msd_fit = np.asarray(msd)[fit_start:]
    popt, pcov = curve_fit(power_law, t_fit, msd_fit, p0=[1.0, 1.0])
    perr = np.sqrt(np.diag(pcov))
    return float(popt[0]), float(popt[1]), float(perr[0]), float(perr[1])


# ==============================================================================
# == Fractal dimension via the uncertainty exponent method
# ==============================================================================

def uncertainty_exponent_fractal_dimension(basin_grid, pixel_size, max_shift_pixels=None, d=2):
    """
    Grid-based estimate of the basin boundary's fractal dimension via the
    uncertainty exponent method (Grebogi et al. 1983).

    The textbook method perturbs each point by a random offset of size eps
    and re-simulates to see whether it lands in a different basin, for a
    range of eps values, then fits f(eps) ~ eps^alpha to the "uncertain"
    fraction. This implementation instead uses the ALREADY-COMPUTED basin
    grid's own spacing as the perturbation: for pixel shifts
    k = 1, 2, 4, 8, ..., it compares each point's basin to its neighbors k
    pixels to the right/down/diagonal, at eps = k * pixel_size.

    Trade-off: this can't probe scales smaller than one grid cell, so the
    fractal dimension estimate is only as reliable as the grid resolution
    allows -- run the escape/basin calculation at high enough resolution
    for the eps range you care about. A rigorous continuum estimate (true
    random sub-pixel perturbations, re-simulated through the CUDA solvers)
    would need the solvers to accept an arbitrary point list rather than a
    regular grid, which they don't yet -- a reasonable future extension if
    this grid-based estimate isn't precise enough.

    pixel_size: physical spacing between adjacent grid cells, e.g.
    (grid_max[i]-grid_min[i])/(grid_dims[i]-1).
    d: dimensionality of the scanned initial-condition slice (2 for a
    standard (x0,y0) scan).

    Returns (D, alpha, eps_values, f_values) -- the last two for
    inspection/plotting the log-log fit.
    """
    nx, ny = basin_grid.shape
    if max_shift_pixels is None:
        max_shift_pixels = min(nx, ny) // 4
    shifts = [2 ** k for k in range(int(np.log2(max(1, max_shift_pixels))) + 1)]

    eps_values, f_values = [], []
    for k in shifts:
        if k >= nx or k >= ny:
            break
        center = basin_grid[:-k, :-k]
        right = basin_grid[:-k, k:]
        down = basin_grid[k:, :-k]
        diag = basin_grid[k:, k:]
        uncertain = (center != right) | (center != down) | (center != diag)
        f = float(np.mean(uncertain))
        if f > 0.0:
            eps_values.append(k * pixel_size)
            f_values.append(f)

    if len(eps_values) < 2:
        raise ValueError("Not enough valid scales to fit an uncertainty exponent "
                          "(grid too small, or basin grid has no boundary in range)")

    log_eps = np.log(eps_values)
    log_f = np.log(f_values)
    alpha, _const = np.polyfit(log_eps, log_f, 1)
    D = d - alpha
    return float(D), float(alpha), np.array(eps_values), np.array(f_values)


# ==============================================================================
# == Convenience wrapper
# ==============================================================================

def analyze_basin(h5_file, basin_dset_name, box_size=5, pixel_size=None, d=2,
                   wada_num_pairs=None, compute_wada=True, compute_fractal_dim=True):
    """
    Loads a basin dataset from h5_file and computes area fractions, basin
    entropy, basin boundary entropy, (optionally) the Wada merging test,
    and (optionally, if pixel_size is given) the fractal dimension.
    Returns a dict of results.
    """
    import h5py
    with h5py.File(h5_file, "r") as f:
        basin = f[basin_dset_name][:]
    basin = np.squeeze(basin)
    if basin.ndim != 2:
        raise ValueError(f"Expected a 2D basin grid after squeezing singleton dims, "
                          f"got shape {basin.shape} from dataset '{basin_dset_name}'")

    results = {
        "shape": basin.shape,
        "area_fractions": basin_area_fractions(basin),
        "basin_entropy": basin_entropy(basin, box_size),
    }
    sbb, n_boundary, n_total = basin_boundary_entropy(basin, box_size)
    results["basin_boundary_entropy"] = sbb
    results["boundary_box_fraction"] = n_boundary / n_total if n_total else 0.0

    if compute_wada and len(np.unique(basin)) >= 3:
        results["wada_merging_test"] = wada_merging_test(basin, box_size, wada_num_pairs)

    if compute_fractal_dim and pixel_size is not None:
        D, alpha, eps, fvals = uncertainty_exponent_fractal_dimension(basin, pixel_size, d=d)
        results["fractal_dimension"] = D
        results["uncertainty_exponent"] = alpha
        results["_uncertainty_exponent_fit"] = (eps, fvals)  # for plotting, if wanted

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <h5_file> <basin_dataset_name> [pixel_size]")
        sys.exit(1)
    h5_file, dset = sys.argv[1], sys.argv[2]
    pixel_size = float(sys.argv[3]) if len(sys.argv) > 3 else None
    results = analyze_basin(h5_file, dset, pixel_size=pixel_size)
    for k, v in results.items():
        if k == "_uncertainty_exponent_fit":
            continue
        print(f"{k}: {v}")
