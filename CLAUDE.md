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
  Compute capability **8.0** → CUDA arch flag should be `sm_80`, NOT `sm_86`
  (the Makefile's current default `sm_86` targets consumer Ampere/RTX 30-series,
  not the A30; it happens to still work since PTX for sm_80 is forward-compatible
  within Ampere, but sm_80 is the correct/native target).
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

## HDF5 / HighFive

- **HDF5** is installed via Debian packages (`libhdf5-dev` and friends), the
  "serial" (non-MPI) variant:
  - Headers: `/usr/include/hdf5/serial`
  - Libs: `/usr/lib/x86_64-linux-gnu/hdf5/serial/` (e.g. `libhdf5.so`) and also
    `/usr/lib/x86_64-linux-gnu/libhdf5_serial.so`.
  - This is **not** at the Makefile's default `HDF5_ROOT := /usr/local/hdf5`
    (that path doesn't exist on this machine).
- **HighFive is NOT installed anywhere on this server.** Searched
  `/usr/local`, `/usr/include`, `/opt`, apt package lists, and `$HOME` — no
  `HighFive/H5File.hpp` found, no matching apt package. This is why `make`
  currently fails at the very first `#include <highfive/H5File.hpp>`.
  HighFive is header-only (MIT license); it needs to be vendored or fetched
  before the build can work. Network access from this host to github.com
  works (verified with `curl`), so `git clone` is an option, but **ask the
  user first** before fetching/installing anything.

## Repo structure (as of Phase 0 inspection)

- `cuda_dynamics_lib/include/cuda_dynamics.h` — core header: `GridSetup`,
  HDF5 save helpers, RK4 stepper (`rk4_step_t`), small device vector/matrix
  helpers. Pulls in all solver headers below.
- `cuda_dynamics_lib/include/solvers/*.cuh` — header-only solvers, all used:
  - `ode_escape.cuh`, `ode_lyapunov.cuh`, `ode_msd.cuh`, `ode_strobo.cuh` — for
    continuous-time (RK4-integrated) systems like `HortonSystem`.
  - `map_escape.cuh`, `map_lyapunov.cuh`, `map_msd.cuh` — for discrete maps
    like `StandardMap`.
- `cuda_dynamics_lib/src/*.cu` (`escape_solver.cu`, `lyapunov_solver.cu`,
  `msd_solver.cu`) — **dead code**: byte-for-byte duplicates of the map-solver
  logic now living in the `.cuh` headers (plus an explicit template
  instantiation for `StandardMap` at the bottom of each). Nothing in the
  Makefile compiles/links these (`LIB_SRC`/`LIB_OBJ` are computed but never
  used in any target's recipe). Superseded by the header-only versions.
- `maps/horton.h` — `HortonSystem`/`HortonSystemParams` (three-wave drift ODE)
  + `SystemTraits<HortonSystem>` (periodic wrap in y, escape in x). Current,
  actively used by `main_horton_escape.cu` and `main_horton_msd.cu`.
- `maps/standard_map.h` — `StandardMap` discrete map + `MapTraits<StandardMap>`.
  Used only by the `.bkp` drivers and the dead `src/*.cu` files; no live
  Makefile target builds it right now.
- `main.cu` — **stale**: references `ThreeWaveSystem`/`ThreeWaveSystemParams`,
  which do not exist anywhere in the repo (renamed to `HortonSystem` at some
  point and `main.cu` never updated). Not part of `make all`, only reachable
  via the unused `dynamics_simulator` target. Needs a decision from the user
  (fix as a stroboscopic-map driver for `HortonSystem`, or delete).
- `main_horton_escape.cu`, `main_horton_msd.cu` — the two live drivers built
  by `make all` (targets `horton_escape`, `horton_msd`).
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

## Build status at Phase 0

`make` (default target `all` → `horton_msd horton_escape`) fails immediately:
`fatal error: highfive/H5File.hpp: No such file or directory`, because
`HDF5_ROOT`/`HIGHFIVE_ROOT` in the Makefile point at paths that don't exist on
this server. Root cause is two-fold: (1) Makefile paths need to be
overridable and pointed at the real HDF5 location, and (2) HighFive itself is
missing and needs to be installed/vendored — see above.
