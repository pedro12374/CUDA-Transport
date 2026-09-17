# CUDA-Transport

A generic CUDA library for transport and chaos analysis of dynamical
systems: escape times and basins, mean-squared displacement, Lyapunov
exponents, and stroboscopic/Poincare maps, all GPU-accelerated and driven
by plain-text config files rather than per-experiment code.

**Generic** means what it says: the library doesn't know or care whether
your system is a three-wave plasma drift model, a driven pendulum, or
something you haven't invented yet. Defining a system is one header file
(the equations of motion, an escape criterion, and how to read its
parameters from a config); running an analysis on it is a config file, not
new code. See [TUTORIAL.md](TUTORIAL.md) for a complete worked example
building a new system from scratch.

## Systems included

| System | Header | What it is |
|---|---|---|
| Horton (three-wave drift) | `maps/horton.h` | A 2D flow forced by up to three waves; a drift-wave transport model. |
| Standard Map | `maps/standard_map.h` | The Chirikov standard map, modified so momentum accumulates as a transport observable rather than staying bounded. |
| Henon-Heiles | `maps/henon_heiles.h` | The textbook Hamiltonian escape system (Henon & Heiles, 1964) -- 4D phase space, fractal 3-basin escape structure above the critical energy. |
| Pendulum (tutorial example) | `maps/pendulum.h` | A driven, damped pendulum; built from scratch in TUTORIAL.md. |

Each ships with a config-driven entry point (`run_horton`, `run_standard_map`,
`run_henon_heiles`) and example configs under `configs/`.

## Dependencies

- **CUDA Toolkit** (`nvcc`), with a compute capability matching your GPU.
- **HDF5**, built with the C++ API (e.g. `libhdf5-dev` on Debian/Ubuntu;
  any recent version).
- A **C++17** host compiler (e.g. `g++` 9+).
- **CMake >= 3.18** (optional -- see "Build" below; the Makefile has no
  extra requirements beyond the above).
- For the Python analysis/plotting scripts (`Py/`): Python 3 with `numpy`,
  `scipy`, `h5py`, and `matplotlib`.

No other libraries are required -- config files use a small dependency-free
parser (`cuda_dynamics_lib/include/config.h`), not JSON/TOML.

## Build

Either build system works; they're kept in sync.

**Makefile** (paths default to a common Debian/Ubuntu layout; override as needed):
```bash
make HDF5_INC=/path/to/hdf5/include HDF5_LIB=/path/to/hdf5/lib CUDA_ARCH=sm_86
```
`CUDA_ARCH` is your GPU's compute capability (see
[developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) --
e.g. `sm_80` for A100/A30, `sm_86` for RTX 30-series, `sm_89` for RTX
40-series). Run with no arguments to use the defaults in the Makefile.

**CMake**:
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86   # your GPU's compute capability
make -j
```
`find_package(HDF5)` locates HDF5 automatically on most systems; pass
`-DHDF5_ROOT=/path/to/hdf5` if it doesn't find yours.

Both produce `run_horton`, `run_standard_map`, `run_henon_heiles` (plus
`run_pendulum` if you've followed the tutorial).

## Quick start

```bash
make run_horton
./run_horton configs/horton_escape.cfg
```
This runs a 1024x1024 escape-basin scan across a 4x3 grid of wave
amplitudes (see `configs/horton_escape.cfg`) and writes
`dat/horton_escape_A2_A3.h5`. (For an instant smoke test instead of the
full run, copy the config and shrink `grid_dims` to e.g. `16, 16` and
`final_time` to `1.0`.)

Plot it:
```python
import sys
sys.path.insert(0, "Py")
import plotting_lib as pl

pl.generate_individual_matrix_plots(
    h5_file="dat/horton_escape_A2_A3.h5",
    output_dir="plots",
    plot_type="basin",
    dset_prefix="EscapeBasin",
    row_params=[0.0, 0.1, 0.5],
    col_params=[0.0, 0.1, 0.5, 1.0],
    row_prefix="A3",
    col_prefix="A2",
)
```

Quantify the basins (entropy, area fractions, fractal dimension, a
Wada-property test -- see `Py/basin_metrics.py`'s module docstring for the
methods and references):
```python
import basin_metrics as bm
import h5py

with h5py.File("dat/horton_escape_A2_A3.h5", "r") as f:
    basin = f["EscapeBasin_A2_0.5000_A3_0.1000"][:]

print(bm.basin_area_fractions(basin))
print(bm.basin_entropy(basin))
```

## Running an analysis: `calculation` types and config keys

Set in a config file's `calculation` key. ODE systems (`run_horton`,
`run_pendulum`): `escape`, `msd`, `lyapunov`, `stroboscopic`. Discrete-map
systems (`run_standard_map`): `escape`, `msd`, `lyapunov`, `phasespace`.
Every calculation needs `output_file` and a grid (`grid_dims`/`grid_min`/
`grid_max`); ODE systems need `dt` and `final_time`; map systems need
`iterations`. See `run_ode_generic()`/`run_map_generic()`'s Doxygen
comments in `cuda_dynamics_lib/include/runner.cuh` for the full list of
keys each calculation type reads, and `configs/*.cfg` for worked examples
of each.

Any config value can be a comma-separated list to sweep a parameter (e.g.
`A2 = 0.0, 0.5, 1.0`) -- every combination is run and saved to the same
HDF5 file, one dataset per combination.

## Testing

```bash
make test        # or: cd build && ctest, if using CMake
```
Runs in a couple of seconds once built. See `tests/README.md` for what's
covered: GPU-vs-independent-CPU-reference trajectories, RK4 convergence
order, confined regular orbits, analytically-known conserved quantities,
and a regression test for the batched long-run solvers.

## Documentation

- [TUTORIAL.md](TUTORIAL.md) -- add a new system from scratch, step by step.
- Doxygen comments throughout the public headers (`cuda_dynamics_lib/include/`,
  `maps/`); `maps/horton.h`'s file-level comment documents the complete
  interface a system implements. Generate HTML docs with `doxygen
  Doxyfile` (requires Doxygen, not bundled).
- `tests/README.md` -- what the test suite checks and why.
- `CLAUDE.md` -- environment notes and project/design history for anyone
  (human or AI) picking this repo back up; not required reading to use the
  library, but useful context on *why* things are built the way they are.

## License

MIT -- see [LICENSE](LICENSE).
