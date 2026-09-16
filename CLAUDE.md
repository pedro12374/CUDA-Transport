# CLAUDE.md — environment and repo notes

Findings from inspecting the server this repo lives on (`swinney`, reached over SSH).
Re-check anything time-sensitive (GPU load, package versions) if it's been a while.

## Machine

- Host: `swinney`, Debian GNU/Linux 11 (bullseye). **Shared machine, no job
  scheduler** (`sinfo`/`module` not found, `run_on_machine.sh` is an empty file).
  Other users run jobs on this box directly — GPU was at 99% util from other
  users' python processes during inspection (2026-09-15).
- `$HOME` (`/home/haerter`) is an **NFS mount** (`henon.if.usp.br:/export/home/haerter`,
  9.1T, 18% used). Recursive `find` over `/` or large subtrees is slow — scope
  searches narrowly (`/usr/local`, `/opt`, specific dirs) rather than scanning from `/`.
- GPU: **1x NVIDIA A30**, 24GB, driver 470.256.02, CUDA driver version 11.4.
  Compute capability **8.0** → Makefile's `CUDA_ARCH` now defaults to `sm_80`
  (was `sm_86`, which targets consumer Ampere/RTX 30-series, not the A30).
- **Never launch long GPU runs here** — it's shared and has no scheduler. Use
  tiny grids (e.g. 16x16) and few steps for any test/verification run. If a
  real run is needed, stop and hand the user the exact command to run inside
  `tmux` themselves.

## Toolchain

- `nvcc`: `/usr/bin/nvcc`, release **11.2**, V11.2.152 (Feb 2021). This is an
  old CUDA Toolkit — max supported host compiler is roughly gcc 10.
- `g++`: `/usr/bin/g++` 10.2.1 (Debian 10.2.1-6) — compatible with nvcc 11.2.
- No `module` system, no alternate toolkits on `PATH`. There is an unrelated
  `/opt/nvidia/hpc_sdk` (NVIDIA HPC SDK) installed but not on `PATH` and not
  used by this project; ignore it unless asked.

## HDF5

- Uses the **HDF5 C++ API directly** (`H5Cpp.h`), not HighFive — HighFive
  isn't installed anywhere on this server (checked `/usr/local`,
  `/usr/include`, `/opt`, apt package lists, `$HOME`) and the user decided to
  drop the dependency rather than vendor/install it.
- HDF5 comes from Debian packages (`libhdf5-dev` and friends), the "serial"
  (non-MPI) variant:
  - Headers: `/usr/include/hdf5/serial` (includes `H5Cpp.h`)
  - Libs: `/usr/lib/x86_64-linux-gnu/hdf5/serial/` — `libhdf5.so` and
    `libhdf5_cpp.so`, linked as `-lhdf5 -lhdf5_cpp`.
  - The Makefile's `HDF5_INC`/`HDF5_LIB` variables default to these paths and
    are overridable (`?=`) for other machines.
- `save_to_h5`/`save_displacement_components` in `cuda_dynamics.h` were
  rewritten against `H5::H5File`/`H5::DataSet`/`H5::DataSpace`, preserving
  the "open existing file and append a dataset, or create fresh" behavior and
  the same dataset names/shapes. Verified with a throwaway 16x16-grid,
  100-step smoke run read back through `h5py` — dataset names, shapes
  (`(16,16)`, `(16,16,2)`, `(100,)`), and value ranges all matched
  expectations (Phase 1, 2026-09-15).

## Repo structure (as of Phase 0 inspection)

- `cuda_dynamics_lib/include/cuda_dynamics.h` — core header: `GridSetup`,
  HDF5 save helpers, RK4 stepper (`rk4_step_t`), small device vector/matrix
  helpers. Pulls in all solver headers below.
- `cuda_dynamics_lib/include/solvers/*.cuh` — header-only solvers, all used:
  - `ode_escape.cuh`, `ode_lyapunov.cuh`, `ode_msd.cuh`, `ode_strobo.cuh` — for
    continuous-time (RK4-integrated) systems like `HortonSystem`.
  - `map_escape.cuh`, `map_lyapunov.cuh`, `map_msd.cuh` — for discrete maps
    like `StandardMap`.
- `maps/horton.h` — `HortonSystem`/`HortonSystemParams` (three-wave drift ODE)
  + `SystemTraits<HortonSystem>` (periodic wrap in y, escape in x). Current,
  actively used by `main_horton_escape.cu` and `main_horton_msd.cu`.
- `maps/standard_map.h` — `StandardMap` discrete map + `MapTraits<StandardMap>`.
  Used only by the `.bkp` drivers; no live Makefile target builds it right now.
- `main_horton_escape.cu`, `main_horton_msd.cu` — the two live drivers built
  by `make all` (targets `horton_escape`, `horton_msd`). Both take an
  optional first CLI argument overriding `FINAL_TIME` (default 1e5), e.g.
  `./horton_msd 1e4`.
- `main_Escape.cu.bkp`, `main_PS.cu.bkp` — backup drivers for `StandardMap`
  (CPU-side escape-time and phase-space calculation via
  `calculate_escape_time`/`calculate_phase_space`, not currently wired into
  the Makefile).
- `Py/` — current plotting stack: `plotting_lib.py` (all the `_plot_*`
  helpers + mosaic/matrix/individual plot generators), `run_plots.py` (driver
  script, reads from `../dat/*.h5`), `parana_theme.py` (color theme),
  `Quant.jl` (Julia, basin-entropy calculation via `Attractors.jl`, reads
  basin H5 datasets written by the C++ code).
- `Presentation/` — manim-slides talk (`press.py`), a quick zoom plot script
  (`plt.py`), and `parana_theme.py` which is byte-identical to `Py/parana_theme.py`
  (duplicated file).
- `plots/` — checked-in PDF outputs (not code).

## Build status

As of Phase 1, `make` (default target `all` → `horton_msd horton_escape`)
builds both drivers cleanly with zero warnings under the default flags. See
git log on the `cleanup` branch for what changed (HighFive → HDF5 C++ API,
Makefile paths/deps/clean target, driver include/buffer/message fixes).

Two things flagged to the user during Phase 1 are now resolved:
`main.cu` (stale `ThreeWaveSystem` references) and the dead
`cuda_dynamics_lib/src/*.cu` duplicates were both deleted with the user's
explicit go-ahead.

## Phase 2 (correctness review) — what changed

- **Fixed, no behavior change**: int-overflow risk in the `ode_*.cuh`
  per-thread index (`int` → `long long`, matching how `map_*.cuh` already
  did it via grid-stride loops), and missing `cudaGetLastError()` checks
  after two kernel launches.
- **Fixed, behavior change (approved by user)**: `ode_strobo.cuh` no longer
  drifts off the nominal `p*tau` clock when `tau` isn't an exact multiple of
  `dt` — it now takes a final partial RK4 step to land exactly on `tau`.
  Not used by any current driver (only `main.cu`, deleted in Phase 1, called
  it), but part of the generic library surface.
- **Batching (approved by user)**: `calculate_ode_escape` and
  `calculate_ode_msd_and_displacement` used to run the entire step loop
  (up to ~1e7 steps × ~1e6 particles = ~1e13 RK4 evaluations) inside one
  kernel launch — bad for a shared, scheduler-less GPU. Both now loop on the
  host in batches (`steps_per_batch`, default 20000), persisting per-particle
  state in device memory between launches. Verified bit-identical (escape)
  / matching-to-FP-noise (MSD, via non-associative `atomicAdd` ordering)
  against the old single-launch behavior.
- **MSD output size (approved by user)**: `calculate_ode_msd_and_displacement`
  now records ~300 logarithmically-spaced samples instead of one value per
  step (was 80MB/pair at 1e7 steps). The physical sample times are returned
  separately and `main_horton_msd.cu` saves them once as a shared
  `MSD_sample_times` HDF5 dataset (same for every (A2,A3) pair) instead of
  duplicating a time axis per pair. `Py/plotting_lib.py`'s `_plot_msd` was
  updated to read this dataset instead of assuming uniform `0.01` spacing.
- **`FINAL_TIME`**: made CLI-overridable in both drivers (optional first
  argument), default unchanged at `1e5`.
- Validated the core integrator itself with a GPU-vs-independently-written-
  CPU-RK4 comparison (8 ICs × 50 steps through the real stroboscopic
  solver): max difference 2.7e-15.

## Python environment note

`Py/plotting_lib.py` currently fails to import in this user's shell:
apt's `scipy` (1.6.0, from `/usr/lib/python3/dist-packages`) references
`numpy.Inf`, which is gone in the `~/.local`-pip-installed `numpy` (2.0.2)
that shadows the system one. This is pre-existing and unrelated to any
change made here — flagged but not fixed (would mean upgrading `scipy` via
`pip install --user --upgrade scipy` or similar, which affects the user's
personal Python environment and wasn't asked for).

## Project intent (see also memory: `project-generic-tool-goal`)

This library is meant to be a **generic** CUDA tool for escape-basin/transport
analysis — Horton and the Standard Map are test systems, not the point.
Prefer fixes that preserve the "define a system + escape criterion, then run
the existing solvers" plug-in architecture over Horton-specific special-casing.
