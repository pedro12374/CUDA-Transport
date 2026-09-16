# Tutorial: adding a new system

This walks through everything this library needs to analyze a **brand new**
dynamical system, end to end: define the equations, write a 4-line driver,
write a config file, run it on the GPU, plot the result, and quantify the
escape basin. Every command and number below was actually run while writing
this tutorial (not just written out) — you should be able to follow along
and reproduce them exactly.

The example: a **driven, damped pendulum**,
```
d(theta)/dt = p
dp/dt = -gamma*p - sin(theta) + A*cos(omega_drive*t)
```
the standard "chaotically driven pendulum" model (see e.g. Strogatz,
*Nonlinear Dynamics and Chaos*). `theta` is the angle from the downward
vertical, `p` its angular velocity, `gamma` damping, and `A`/`omega_drive`
the amplitude/frequency of a periodic driving torque. It's a good tutorial
system because it's genuinely different from the three systems already in
the library (Horton, Standard Map, Henon-Heiles) but still simple enough to
derive by hand in a few lines, and it has a natural, physically meaningful
escape criterion: does the pendulum stay trapped oscillating near the
bottom, or does it go over the top and start rotating?

**Prerequisites**: you've built the library once already (`make`, see
[README.md](README.md) for dependencies), so `cuda_dynamics_lib/` and
`maps/` are on your include path via the Makefile/CMake as-is.

## Step 1: Define the system

Every system needs four things, all in one header under `maps/`. The
cleanest way to see this pattern is to read [`maps/horton.h`](maps/horton.h)
(its file-level comment documents the full interface contract this
tutorial follows) or [`maps/standard_map.h`](maps/standard_map.h) for the
discrete-map equivalent — but we'll build ours from scratch here.

Create `maps/pendulum.h`:

```cpp
#pragma once
#include <cmath>
#include "config.h"

template <typename SystemType>
struct SystemTraits;

// 1. The parameter struct: whatever your equations need, one field each.
struct PendulumParams {
    double gamma;
    double A;
    double omega_drive;
};

// 2. The system functor: the right-hand side of dstate/dt = f(state, t),
//    plus (optionally) its Jacobian for the Lyapunov solver.
struct PendulumSystem {
    template <int DIMS>
    __host__ __device__ void operator()(
        const double state[DIMS],
        double dstate_dt[DIMS],
        const PendulumParams& params,
        double t) const
    {
        double theta = state[0];
        double p = state[1];
        dstate_dt[0] = p;
        dstate_dt[1] = -params.gamma * p - sin(theta) + params.A * cos(params.omega_drive * t);
    }

    template <int DIMS>
    __host__ __device__ void jacobian(
        const double state[DIMS],
        const PendulumParams& params,
        double J_out[DIMS * DIMS],
        double /*t*/) const
    {
        double theta = state[0];
        J_out[0] = 0.0;             // d(dtheta/dt)/dtheta
        J_out[1] = 1.0;             // d(dtheta/dt)/dp
        J_out[2] = -cos(theta);     // d(dp/dt)/dtheta
        J_out[3] = -params.gamma;   // d(dp/dt)/dp
    }
};

// 3. The trait specialization: boundary handling + escape criterion.
template <>
struct SystemTraits<PendulumSystem> {
    // theta is left unwrapped on purpose (see below) -- no periodic
    // wrapping, so this is a no-op.
    __host__ __device__ static void post_step_update(double /*state*/[2]) {}

    __host__ __device__ static int check_escape(const double state[2]) {
        const double N_ROTATIONS = 3.0;
        const double THRESHOLD = N_ROTATIONS * 2.0 * M_PI;
        double theta = state[0];
        if (theta > THRESHOLD) return 1;
        if (theta < -THRESHOLD) return -1;
        return 0;
    }
};

// 4. Build the params struct from a config file.
template <>
inline PendulumParams load_params_for<PendulumParams>(const Config& cfg) {
    PendulumParams p;
    p.gamma = cfg.get_double("gamma", 0.5);
    p.A = cfg.get_double("A", 1.5);
    p.omega_drive = cfg.get_double("omega_drive", 0.666667);
    return p;
}
```

A few things worth noticing:

- **`operator()` takes `t`, and uses it.** Horton and Henon-Heiles both take
  `t` (the RK4 stepper always passes it), but only Horton actually uses it
  (its forcing waves move in time); Henon-Heiles ignores it. Here, the
  driving torque `A*cos(omega_drive*t)` genuinely depends on it.
- **`theta` is deliberately never wrapped into `[-pi,pi]`.** This is the
  same choice `maps/standard_map.h` makes for its momentum coordinate: by
  letting `theta` accumulate without bound, `check_escape()` can read off
  "how many net rotations has this pendulum completed" directly from its
  value, rather than needing to track a separate unwrapped copy. If you
  *did* want `theta` wrapped (e.g. to plot a phase portrait on a cylinder),
  you'd `fmod` it into range here — see `SystemTraits<HortonSystem>::
  post_step_update` in `maps/horton.h` for a worked example of that instead.
- **The escape criterion is a judgment call, and it's yours to make.**
  "3 full rotations" isn't derived from anything — it's a reasonable,
  round-number choice for "clearly rotating, not just briefly kicked over
  the top and back." Your own systems will have their own natural
  criteria (see `maps/henon_heiles.h`'s radius-based one for a system
  where the escape region is a literal spatial region, or
  `maps/horton.h`'s domain-boundary one).

## Step 2: Write the entry point

This part almost never changes shape — it's the same ~4 lines for any ODE
system. Create `run_pendulum.cu`:

```cpp
#include "cuda_dynamics_lib/include/cuda_dynamics.h"
#include "cuda_dynamics_lib/include/runner.cuh"
#include "maps/pendulum.h"

int main(int argc, char** argv) {
    return run_ode_generic<2, PendulumSystem, PendulumParams>(argc, argv);
}
```

The `2` is `DIMS` — the state dimensionality, fixed at compile time (this
is the one place you tell the library "how big is my state vector"; see
`run_ode_generic`'s docs in `runner.cuh` for why it can't be a runtime
choice). For a discrete map instead of an ODE, you'd call
`run_map_generic<DIMS, YourMap, YourMapParams>(argc, argv)` instead — see
`run_standard_map.cu`.

## Step 3: Add it to the build

**Makefile**, add a rule (and, if you want it built by default, add its
name to the `all:` target):
```makefile
run_pendulum: run_pendulum.cu $(HEADERS)
	@echo "==> Building Pendulum generic runner"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
```

**CMakeLists.txt**, add one line next to the other `add_cudat_driver` calls:
```cmake
add_cudat_driver(run_pendulum run_pendulum.cu)
```

Build it:
```bash
make run_pendulum
```

## Step 4: Write a config file

Create `configs/pendulum_escape.cfg`:
```
calculation = escape

dt = 0.01
final_time = 100

grid_dims = 200, 200
grid_min = -3.14159265358979, -2.0
grid_max = 3.14159265358979, 2.0

output_file = dat/pendulum_escape.h5

gamma = 0.5
A = 1.0, 1.5, 2.0
omega_drive = 0.666667
```

`grid_dims`/`grid_min`/`grid_max` scan the initial condition
`(theta0, p0)` over a 200x200 grid; `A = 1.0, 1.5, 2.0` is a **parameter
sweep** (any comma-separated value is one — see `config.h`'s docs) across
three driving amplitudes spanning the system's transition to chaos, run as
three separate escape-basin calculations into one HDF5 file. `gamma` and
`omega_drive` are fixed (no comma).

## Step 5: Run it

```bash
./run_pendulum configs/pendulum_escape.cfg
```
```
run_ode_generic: 3 run(s), calculation='escape', 40000 particles

--- [1/3] A_1.0000 ---
Saved 'escape' results for A_1.0000.

--- [2/3] A_1.5000 ---
Saved 'escape' results for A_1.5000.

--- [3/3] A_2.0000 ---
Saved 'escape' results for A_2.0000.

All 'escape' runs complete.
```
This took well under a second on an A30 for all 40,000 x 3 particles. The
result: 0% of particles escape at `A=1.0`, 75.8% at `A=1.5` (deep in the
chaotic regime), 35.3% at `A=2.0` — you can check this yourself:
```python
import h5py, numpy as np
with h5py.File("dat/pendulum_escape.h5", "r") as f:
    for a in ["1.0000", "1.5000", "2.0000"]:
        t = f[f"EscapeTime_A_{a}"][:]
        print(a, (t != -1).mean())
```

## Step 6: Plot the results

`Py/plotting_lib.py` already knows how to plot an escape-basin dataset —
you don't need to write any new plotting code:
```python
import sys
sys.path.insert(0, "Py")
import plotting_lib as pl

pl.generate_individual_plots(
    h5_file="dat/pendulum_escape.h5",
    output_dir="plots",
    plot_type="basin",
    dset_prefix="EscapeBasin_A",
    params_list=[1.0, 1.5, 2.0],
    title_prefix="Pendulum escape basins",
    bounds=[-3.14159, 3.14159, -2.0, 2.0],
)
```
This writes `plots/basin_EscapeBasin_A_1.0000.pdf` (and `_1.5000`,
`_2.0000`). The `A=1.5` one shows a genuinely fractal basin boundary —
the classic signature of chaotic escape dynamics, not an artifact.

(If your local LaTeX install is incomplete, `Py/parana_theme.py` sets
`usetex=True` and plotting will fail — see `CLAUDE.md`'s Python
environment notes.)

## Step 7 (bonus): quantify the basin

`Py/basin_metrics.py` (see its module docstring for the underlying methods)
works on any basin dataset, including this new one:
```python
sys.path.insert(0, "Py")
import basin_metrics as bm
import h5py

with h5py.File("dat/pendulum_escape.h5", "r") as f:
    basin = f["EscapeBasin_A_1.5000"][:]

print("area fractions:", bm.basin_area_fractions(basin))
print("basin entropy:", bm.basin_entropy(basin))
sbb, n_boundary, n_total = bm.basin_boundary_entropy(basin)
print("boundary entropy:", sbb)

pixel_size = (3.14159265358979 * 2) / 199  # (grid_max - grid_min) / (grid_dims - 1), theta axis
D, alpha, eps, f_eps = bm.uncertainty_exponent_fractal_dimension(basin, pixel_size)
print("fractal dimension:", D)
```
No new code needed — `basin_metrics.py` only ever looks at the HDF5 array,
not at how it was produced.

## Step 8 (bonus): add a regression test

If this system is going to stick around, it's worth a
`tests/test_gpu_vs_cpu.cu`-style check. The pattern (see that file and
`tests/common.h`): build a handful of hand-picked initial conditions, run
them through the real GPU solver, and compare against
`cpu_rk4_step<DIMS, PendulumSystem, PendulumParams>(...)` — an
independently-written CPU RK4 stepper already provided by `tests/common.h`,
so you don't need to write your own reference integrator.

## Recap

You wrote one header (the physics), one 4-line entry point, one Makefile
rule and one CMake line, and one config file. Nothing in
`cuda_dynamics_lib/` changed. From here:
- [README.md](README.md) — dependencies, full build instructions, the three
  systems shipped with the library.
- `maps/horton.h`'s file-level Doxygen comment — the complete, formal
  interface contract every system (including this tutorial's) implements.
- `CLAUDE.md` — this server's environment notes, and the design history
  behind the generic runner, if you want the "why" behind the "how" above.
- `tests/README.md` — what the existing test suite checks, if you're
  adding a system you want covered the same way.
