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

## Repo structure

- `cuda_dynamics_lib/include/cuda_dynamics.h` — core header: `GridSetup`,
  HDF5 save helpers, RK4 stepper (`rk4_step_t`), small device vector/matrix
  helpers. Pulls in all solver headers below.
- `cuda_dynamics_lib/include/solvers/*.cuh` — header-only solvers, all used:
  - `ode_escape.cuh`, `ode_lyapunov.cuh`, `ode_msd.cuh`, `ode_strobo.cuh` — for
    continuous-time (RK4-integrated) systems like `HortonSystem`.
  - `map_escape.cuh`, `map_lyapunov.cuh`, `map_msd.cuh` — for discrete maps
    like `StandardMap`.
- `cuda_dynamics_lib/include/config.h` — dependency-free `key = value` config
  file reader (`Config`), plus parameter-sweep support (`enumerate_sweeps`)
  and the `load_params_for<ParamsType>` extension point every system
  specializes. See "Generic library architecture" below.
- `cuda_dynamics_lib/include/runner.cuh` — `run_ode_generic<DIMS,SystemType,
  ParamsType>` / `run_map_generic<DIMS,MapType,ParamsType>`: the actual
  generic driver logic (grid setup, sweep loop, calculation dispatch, HDF5
  output). System-agnostic; never needs editing to add a system.
- `maps/horton.h` — `HortonSystem`/`HortonSystemParams` (three-wave drift ODE)
  + `SystemTraits<HortonSystem>` (periodic wrap in y, escape in x) +
  `load_params_for<HortonSystemParams>`. The reference example of a
  complete ODE system definition.
- `maps/standard_map.h` — `StandardMap`/`StandardMapParams` discrete map +
  `MapTraits<StandardMap>` + `load_params_for<StandardMapParams>`. The
  reference example of a complete discrete-map system definition.
- `maps/henon_heiles.h` — `HenonHeilesSystem`/`HenonHeilesParams` (the
  classic Hamiltonian escape system), `DIMS=4` (x,y,px,py) -- the third
  system, added in Phase 3 partly to prove the generic runner is actually
  generic over DIMS, not just over which 2D system is plugged in. Also
  exposes `henon_heiles_energy` (conserved quantity, used by
  `tests/test_conservation.cu`).
- `run_horton.cu`, `run_standard_map.cu`, `run_henon_heiles.cu` — the
  trivial (~4 line) per-system entry points; see "Generic library
  architecture" below.
- `configs/*.cfg` — example/working config files for all three systems.
- `tests/` — the test suite (`make test`); see `tests/README.md` and the
  "Phase 3" section below.
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

`make` (default target `all` → `run_horton run_standard_map run_henon_heiles`)
builds all three generic drivers cleanly with zero warnings under the
default flags. `make test` builds and runs the test suite (`tests/`, ~2s
once built). See git log on the `cleanup` branch for full history
(HighFive → HDF5 C++ API, Makefile fixes, Phase 2 correctness fixes, the
generic-runner rewrite, Phase 3's test suite + Henon-Heiles).

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
  separately and saved once as a shared `MSD_sample_times` HDF5 dataset
  (same for every sweep combination) instead of duplicating a time axis per
  dataset. `Py/plotting_lib.py`'s `_plot_msd` was updated to read this
  dataset instead of assuming uniform `0.01` spacing.
- Validated the core integrator itself with a GPU-vs-independently-written-
  CPU-RK4 comparison (8 ICs × 50 steps through the real stroboscopic
  solver): max difference 2.7e-15.
- (`FINAL_TIME` was briefly made CLI-overridable on the old drivers; that's
  now superseded by the config file's `final_time` key, which is strictly
  more general — see below.)

## Generic library architecture

Per the user's explicit request (2026-09-16): the point of this repo is a
**generic** transport/escape-basin analysis tool, not a Horton-specific one.
`main_horton_escape.cu`/`main_horton_msd.cu` (one hand-written driver per
system x calculation, with everything hardcoded) were replaced by:

1. **A system header** (`maps/horton.h`, `maps/standard_map.h`) — the
   complete definition of a dynamical system: the functor (`operator()`
   [+ `jacobian` for Lyapunov]), `SystemTraits<T>` (ODE: `post_step_update`,
   `check_escape`) or `MapTraits<T>` (map: `msd_dimension_index`,
   `check_escape`), and a `load_params_for<ParamsType>` specialization that
   builds the params struct from a `Config`. This is the interface a new
   system must implement (candidate content for the TUTORIAL.md in Phase 5).
2. **The generic runner** (`cuda_dynamics_lib/include/runner.cuh`):
   `run_ode_generic<DIMS, SystemType, ParamsType>` and
   `run_map_generic<DIMS, MapType, ParamsType>`. Reads a config file, sets up
   `GridSetup`, dispatches to whichever `calculate_*` solver the config's
   `calculation` key names (`escape`/`msd`/`lyapunov`/`stroboscopic` for ODE;
   `escape`/`msd`/`lyapunov`/`phasespace` for maps), saves HDF5 output.
   Never needs editing to add a system or run a different calculation.
3. **A ~4-line per-system entry point** (`run_horton.cu`,
   `run_standard_map.cu`): `#include` the system header + `runner.cuh`, call
   `run_ode_generic<...>` or `run_map_generic<...>` from `main`. This is the
   one place "is this a map or an ODE" gets decided, by which function is
   called (DIMS is also fixed here, at compile time — templates can't be
   runtime-polymorphic without type erasure, which would cost the
   zero-overhead device-functor design this library already relies on).
4. **Config files** (`configs/*.cfg`) — dependency-free `key = value` text
   (`cuda_dynamics_lib/include/config.h`; no JSON library needed/installed).
   Any value with a comma becomes a parameter sweep axis (cartesian product
   across all swept keys), except `grid_dims`/`grid_min`/`grid_max` which are
   always fixed-length per-dimension vectors. Dataset names get a
   `_key_value_key_value` suffix built from the swept keys in file order --
   `configs/horton_escape.cfg`'s `A2`/`A3` sweep reproduces the exact
   `EscapeTime_A2_..._A3_...` naming the old driver + `Py/plotting_lib.py`
   already expected, so nothing downstream needed to change.

Adding a brand new system to the library: write one header (functor +
traits + `load_params_for`), write one ~4-line entry-point .cu, add one
Makefile rule, write a config file. No solver or runner code touched.

Verified: `configs/horton_escape.cfg` and `configs/horton_msd.cfg` run
through `run_horton` at tiny scale (16x16 grid) produce byte-identical MSD
values to the old `main_horton_msd.cu` driver's Phase 2 smoke-test output --
confirms the rewrite preserved behavior exactly, not just "runs without
crashing." `run_standard_map` + `configs/standard_map_escape.cfg` exercises
the discrete-map path end-to-end as a second worked example.

## Phase 3: test suite + Henon-Heiles

Per the user's request (2026-09-16): built the `tests/` suite (`make test`)
covering the existing systems (Horton, Standard Map) and added a third
system, Henon-Heiles (`maps/henon_heiles.h`) — a paradigmatic Hamiltonian
escape system, `H = 1/2(px^2+py^2) + 1/2(x^2+y^2) + lambda*(x^2*y - y^3/3)`
(`lambda=1` is the textbook form), `DIMS=4` (x,y,px,py). Deliberately chosen
as the third system because it's *not* 2D like Horton/Standard Map — a real
test that the generic runner (`run_ode_generic<DIMS,...>`) is actually
generic over DIMS, not just over which 2D system is plugged in. Escape
criterion: `r > 10`, basin 1/2/3 records which of the three saddle-point
channels (at 90°/210°/330°, matching the classic literature result) the
particle left through. `configs/henon_heiles_escape.cfg` reproduces the
standard "released from rest" (px0=py0=0) fractal escape-basin scan.

Along the way, fixed a real latent bug in `GridSetup` needed to make that
config file work at all: a single-point dimension (`grid_res[j]==1`, used
to hold px0/py0 fixed while scanning x0/y0) divided by `grid_res[j]-1 == 0`,
producing NaN. Now a size-1 dimension just takes `min_b[j]`. No effect on
any existing multi-point dimension.

`tests/` (see `tests/README.md` for the full table) — `make test` runs in
~2s once built (a cold `nvcc` build of all 5 test binaries takes ~45s, but
that's compile time, not test time):
- **GPU vs CPU reference** (`test_gpu_vs_cpu.cu`): Horton and Henon-Heiles
  via the real stroboscopic solver (tau==dt) vs `tests/common.h`'s
  independently-written `cpu_rk4_step`; Standard Map's GPU escape kernel vs
  a host-side map loop. Found and worked around a subtlety, not a bug: at
  chaotic K (tried K=1.5 first), a few ULP of benign host/device
  floating-point difference gets exponentially amplified by the positive
  Lyapunov exponent into a completely different escape time by iteration
  ~180 — confirmed by direct trajectory inspection (p random-walks past
  several multiples of pi). Fixed by using sub-chaotic K=0.5 for this
  specific bit-level check; chaotic-regime behavior is what
  `test_confinement.cu` exercises instead (aggregate escape/no-escape only,
  not per-particle timing).
- **Convergence** (`test_convergence.cu`): RK4 is 4th order, so halving dt
  should cut the error ~16x; checked via dt/dt/2/dt/4 against each other.
  Horton needed dt=0.005 (not 0.02) to actually reach the asymptotic ~16
  ratio, since its effective frequency scale (kx1=6, ky1=3) is higher than
  Henon-Heiles' O(1) — verified empirically by scanning dt before picking
  the test's value (same finding shows up in `test_conservation.cu`).
- **Confinement** (`test_confinement.cu`): sub-critical/regular orbits never
  escape. For the Standard Map, learned (and documented in the test) that
  "K below the chaos threshold" is *not* sufficient on its own — only
  initial conditions well inside a stable island are guaranteed confined;
  particles starting near the |p|=pi escape boundary can still be caught in
  a local stochastic layer and drift across it even at low K. Verified
  empirically which (K, p0-range) combinations are actually safe before
  picking the test's parameters. Henon-Heiles' version is a rigorous
  consequence of energy conservation (E < E_c=1/6 bounds the orbit forever,
  not just empirically observed).
- **Conservation** (`test_conservation.cu`): validates the Phase-3-planning
  question from the original roadmap ("check the equations first and tell
  me if \[the stream-function] test is valid") — confirmed valid by direct
  differentiation: for Horton's single-wave case (A2=A3=0), the flow is
  divergence-free and autonomous, and
  `psi(x,y) = -A1*sin(kx1*x)*cos(ky1*y)` (now in `maps/horton.h` as
  `horton_single_wave_stream_function`) is exactly conserved. Henon-Heiles'
  version is the standard total-energy check (`henon_heiles_energy`, also
  in its header).
- **Batching regression** (`test_batching.cu`): formalizes the Phase 2
  batched-vs-unbatched validation (previously only done ad hoc in scratch)
  into a permanent test, extended to also cover Henon-Heiles.

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
