# Test suite

Run with `make test` from the repo root (or `make -C tests` directly).
Each `test_*.cu` is its own small, self-contained binary with its own
`main()` — see `common.h` for the ~40-line pass/fail framework used
throughout (no external test library) and its independently-written
`cpu_rk4_step`/`cpu_integrate`, used as a from-scratch cross-check against
the real GPU kernels.

All test *execution* finishes in a couple of seconds (compiling the five
`.cu` files from a clean state takes longer, dominated by `nvcc`, not the
tests themselves).

| File | What it checks |
|---|---|
| `test_gpu_vs_cpu.cu` | GPU trajectories match an independent CPU RK4 reference (Horton, Henon-Heiles via the real stroboscopic solver with tau==dt) and GPU escape times match a host-side map iteration (Standard Map, at sub-chaotic K — see the comment in the file for why chaotic K makes a strict per-particle comparison meaningless). |
| `test_convergence.cu` | RK4 is 4th order: halving `dt` should shrink the error ~16x (Horton, Henon-Heiles), checked entirely on the CPU reference stepper. |
| `test_confinement.cu` | Regular/sub-critical orbits never trigger the escape criterion: Standard Map well below the chaos threshold K_c≈0.9716 (restricted to initial `p` near a stable island — see the file for why sub-critical K alone isn't sufficient), and Henon-Heiles below the critical energy E_c=1/6 (verified bounded forever by energy conservation). |
| `test_conservation.cu` | An analytically-known conserved quantity stays (nearly) constant along the numerical trajectory: Horton's single-wave (A2=A3=0) co-moving-frame stream function (derived in `maps/horton.h`), and Henon-Heiles' total energy. |
| `test_batching.cu` | The batched `calculate_ode_escape`/`calculate_ode_msd_and_displacement` (see CLAUDE.md Phase 2) give identical results regardless of `steps_per_batch` — regression test for that refactor. |

Systems covered: Horton (`maps/horton.h`), Standard Map
(`maps/standard_map.h`), Henon-Heiles (`maps/henon_heiles.h`) — exercising
every system currently in the library, both the ODE and discrete-map solver
families, and (via Henon-Heiles) a system with `DIMS != 2`.
